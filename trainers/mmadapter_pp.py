from collections import OrderedDict

import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from .gpt3_prompts import load_CuPL_templates
from .imagenet_templates import IMAGENET_TEMPLATES

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, tk_embeds, tk_prompts, text_adapter_func=None):
        x = tk_embeds + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if text_adapter_func == None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, text_adapter_func])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tk_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class AdapterLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.n_cls = len(classnames)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # build multi-modal adapter
        self.text_adapter_func = lambda x, i: self.return_text_feature(x, i)
        self.text_adapter = self.build_adapter(
            clip_model.ln_final.weight.shape[0], 
            len(clip_model.transformer.resblocks), 
            cfg.TRAINER.MMADAPTERPP.ADAPTER_START,
            cfg.TRAINER.MMADAPTERPP.ADAPTER_END,
            cfg.TRAINER.MMADAPTERPP.ADAPTER_DIM,
            clip_model.dtype,
            cfg.TRAINER.MMADAPTERPP.ADAPTER_DROP
        )
        
        self.visual_adapter_func = lambda x, i: self.return_visual_feature(x, i)
        self.visual_adapter = self.build_adapter(
            clip_model.visual.ln_post.weight.shape[0],
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.MMADAPTERPP.ADAPTER_START,
            cfg.TRAINER.MMADAPTERPP.ADAPTER_END,
            cfg.TRAINER.MMADAPTERPP.ADAPTER_DIM,
            clip_model.dtype,
            cfg.TRAINER.MMADAPTERPP.ADAPTER_DROP
        )

        self.shared_adapter = self.build_adapter(
            cfg.TRAINER.MMADAPTERPP.ADAPTER_DIM,
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.MMADAPTERPP.ADAPTER_START,
            cfg.TRAINER.MMADAPTERPP.ADAPTER_END,
            cfg.TRAINER.MMADAPTERPP.ADAPTER_DIM,
            clip_model.dtype,
            cfg.TRAINER.MMADAPTERPP.ADAPTER_DROP
        )

        self.adapter_scale = cfg.TRAINER.MMADAPTERPP.ADAPTER_TRAIN_SCALE 
        self.adapter_scale_reg = cfg.TRAINER.MMADAPTERPP.ADAPTER_TRAIN_SCALE_REG
        self.adapter_test_scale = cfg.TRAINER.MMADAPTERPP.ADAPTER_TEST_SCALE 

    def return_feature(self, x, adapter, shared_adapter, adapter_scale):
        y = adapter.down(x)
        if shared_adapter is not None:
            y = shared_adapter(y)
        y = adapter.up(y) * adapter_scale
        return y

    def return_text_feature(self, x, index):
        text_adapter = self.text_adapter[index]
        shared_adapter = self.shared_adapter[index]
        if text_adapter == None:
            return 0
        y = self.return_feature(
            x, text_adapter, shared_adapter, 
            self.adapter_scale
        )
        return y

    def return_visual_feature(self, x, index):
        visual_adapter = self.visual_adapter[index]
        shared_adapter = self.shared_adapter[index]
        if visual_adapter == None:
            return 0
        y = self.return_feature(
            x, visual_adapter, shared_adapter, 
            self.adapter_scale
        )
        return y


    def build_adapter(self, d_model, n_layers, l_start, l_end, mid_dim, dtype, drop=0.0):
        adapter = [None] * (n_layers + 1)
        for i in range(l_start, l_end+1):
            if mid_dim == d_model:
                adapter[i] = nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.ReLU(),
                    nn.Dropout(p=drop)
                )
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("down", nn.Sequential(nn.Linear(d_model, mid_dim), nn.ReLU())),
                    ("up", nn.Linear(mid_dim, d_model))
                ]))
        adapter = nn.ModuleList([a for a in adapter])
        for m in adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if dtype == torch.float16:
            for m in adapter.modules():
                m.half()
    
        return adapter
    
    def forward(self):
        return self.text_adapter_func, self.visual_adapter_func

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.adapter_learner = AdapterLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # for inference
        self.text_features_for_inference = None

        # for multiple-prompts
        self.input_prompt_type = cfg.TRAINER.MMADAPTERPP.INPUT_PROMPTS.lower()
        if self.input_prompt_type == "cupl":
            self.tk_prompts = self.build_cupl_prompts(cfg, classnames)
        elif self.input_prompt_type == "hc":
            self.tk_prompts = self.build_hc_prompts(cfg, classnames)
        self.token_embedding = clip_model.token_embedding
        
        # for self-regularization
        self.ssreg = nn.CosineSimilarity(dim=-1)

    def build_cupl_prompts(self, cfg, classnames):
        # Prompt Contexts
        cupl_prompts = load_CuPL_templates(cfg.DATASET.NAME)
        cupl_prompts = {k.lower().replace("_", " "): v for k, v in cupl_prompts.items()}
        classnames = [name.replace("_", " ") for name in classnames]
        tk_prompts = []
        for cname in classnames:
            prompts = cupl_prompts[cname.lower().replace("_", " ")]
            prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts])
            tk_prompts.append(prompts)
        return torch.stack(tk_prompts, dim=0)
    
    def build_hc_prompts(self, cfg, classnames):
        # Prompt Contexts
        classnames = [name.replace("_", " ") for name in classnames]
        tk_prompts = []
        for cname in classnames:
            prompts = [st.replace("{}", cname) for st in IMAGENET_TEMPLATES]
            prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts])
            tk_prompts.append(prompts)
        return torch.stack(tk_prompts, dim=0)
    

    def encode_text(self, tk_embeds, tk_prompts, adapter_func=None):
        if adapter_func is not None:
            text_features = self.text_encoder(tk_embeds, tk_prompts, adapter_func)
        else:
            text_features = self.text_encoder(tk_embeds, tk_prompts)
        return F.normalize(text_features, dim=-1)
    
    def encode_image(self, image, adapter_func=None):
        if adapter_func is not None:
            image_features = self.image_encoder([image.type(self.dtype), adapter_func])
        else:
            image_features = self.image_encoder(image.type(self.dtype))
        return F.normalize(image_features, dim=-1)


    def forward(self, image):
  
        text_adapter_func, visual_adapter_func = self.adapter_learner()

        if self.adapter_learner.training:
            # for main branch forwarding
            n_cls, n_temp = self.tk_prompts.shape[0:2]
            if self.input_prompt_type == "cupl" or self.input_prompt_type == "hc":
                rand_id = torch.randint(0, n_temp, (1, n_cls), dtype=torch.long)
            else:
                rand_id = 0
            tk_prompts = self.tk_prompts[torch.arange(n_cls), rand_id].squeeze(0)

            with torch.no_grad():
                tk_embeds = self.token_embedding(tk_prompts.to(image.device))

            text_features = self.encode_text(tk_embeds, tk_prompts, text_adapter_func)
            image_features = self.encode_image(image, visual_adapter_func)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            # for regularization branch
            init_scale = self.adapter_learner.adapter_scale
            curr_scale = torch.rand(1).to(image.device) * init_scale
            self.adapter_learner.adapter_scale = curr_scale

            with torch.no_grad():
                text_features_reg = self.encode_text(tk_embeds, tk_prompts, text_adapter_func)
                image_features_reg = self.encode_image(image, visual_adapter_func)

            self.adapter_learner.adapter_scale = init_scale
            
            t_loss = self.ssreg(text_features, text_features_reg.detach()).mean()
            i_loss = self.ssreg(image_features, image_features_reg.detach()).mean()

            reg = self.adapter_learner.adapter_scale_reg
            return logits, -t_loss * reg, -i_loss * reg

        else:
            self.adapter_learner.adapter_scale = self.adapter_learner.adapter_test_scale
            if self.text_features_for_inference is None:
                tk_prompts = self.tk_prompts
                n_cls, n_temp = tk_prompts.shape[0:2]
                mean_text_features = 0
                for tid in range(n_temp):
                    tkp = tk_prompts[:, tid].to(image.device)
                    with torch.no_grad():
                        tk_embeds = self.token_embedding(tkp)
                    text_features = self.encode_text(tk_embeds, tkp, text_adapter_func)
                    mean_text_features += text_features
                mean_text_features /= n_temp
                self.text_features_for_inference = F.normalize(mean_text_features, dim=-1)
       
            text_features = self.text_features_for_inference

            image_features = self.encode_image(image, visual_adapter_func)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return logits


@TRAINER_REGISTRY.register()
class MultiModalAdapterPP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMADAPTERPP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.MMADAPTERPP.PREC == "fp32" or cfg.TRAINER.MMADAPTERPP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()


        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        
        for name, param in self.model.named_parameters():
            if "text_adapter" not in name and "visual_adapter" not in name and "shared_adapter" not in name:
                param.requires_grad_(False)

        # Double check
        num_trainable_params = 0
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                num_trainable_params += param.data.nelement()
        print(f"Parameters to be updated: {enabled}")
        print(f"Number of trainable parameters: {num_trainable_params}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.adapter_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter_learner", self.model.adapter_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MMADAPTERPP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.MMADAPTERPP.PREC
        if prec == "amp":
            with autocast():
                output, t_loss, i_loss = self.model(image)
                c_loss = F.cross_entropy(output, label)
                loss = c_loss + t_loss + i_loss
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": c_loss.item(),
            "t_loss": t_loss.item(),
            "i_loss": i_loss.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):

        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_embedding" in state_dict:
                del state_dict["token_embedding"]
            if "tokenized_prompts" in state_dict:
                del state_dict["tokenized_prompts"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)