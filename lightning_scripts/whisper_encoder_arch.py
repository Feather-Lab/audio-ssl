import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as taF
import whisper
import whisper.model as whisper_models

WHISPER_SR = 16_000

def get_whisper_encoder_layer_map(n_layer: int) -> dict:
    layer_map = {
        "input_after_preproc": "input_after_preproc",
        "conv1": "conv1",
        "conv2": "conv2",
        "ln_post": "ln_post",
        "final": "final",
    }
    for idx in range(n_layer):
        layer_map[f"encoder_block_{idx}"] = f"encoder_block_{idx}"
        layer_map[f"block_{idx}"] = f"encoder_block_{idx}"
    return layer_map

def get_whisper_encoder_layer_sizes(encoder_kwargs: dict, time_average: bool) -> dict:
    n_mels = encoder_kwargs.get("n_mels", 80)
    n_ctx = encoder_kwargs.get("n_ctx", 1500)
    n_state = encoder_kwargs.get("n_state", 384)
    n_layer = encoder_kwargs.get("n_layer", 4)

    if time_average:
        base_time_size = n_state
        input_size = n_mels
    else:
        base_time_size = n_state * n_ctx
        input_size = n_mels * n_ctx

    layer_sizes = {
        "input_after_preproc": input_size,
        "conv1": base_time_size,
        "conv2": base_time_size,
        "ln_post": base_time_size,
        "final": base_time_size,
    }
    for idx in range(n_layer):
        layer_sizes[f"encoder_block_{idx}"] = base_time_size
        layer_sizes[f"block_{idx}"] = base_time_size
    return layer_sizes


def whisper_layer_names_list(n_blocks: int) -> list[str]:
    names = [f"encoder_block_{idx}" for idx in range(n_blocks)]
    names.append("ln_post")
    return names


def parse_whisper_layer_str(layer: str, valid_layers: list[str]) -> str:
    if layer in valid_layers:
        return layer
    msg = f"Invalid Whisper layer '{layer}'. Valid layers: {valid_layers}"
    raise ValueError(msg)


class WhisperLayerwiseEncoder(nn.Module):
    """Pretrained Whisper encoder wrapper with per-layer activations."""

    def __init__(self, whisper_model_name: str = "large-v3"):
        super().__init__()
        model = whisper.load_model(whisper_model_name)
        self.encoder = model.encoder
        self.n_mels = model.dims.n_mels
        self.layer_names = whisper_layer_names_list(len(self.encoder.blocks))

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self._layer_outputs: dict[str, torch.Tensor] = {}
        for idx, block in enumerate(self.encoder.blocks):
            block.register_forward_hook(self._make_hook(f"encoder_block_{idx}"))

    def _make_hook(self, name: str):
        def hook_fn(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            self._layer_outputs[name] = out

        return hook_fn

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor, sr: int | None = None) -> dict[str, torch.Tensor]:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        if sr is not None and sr != WHISPER_SR:
            waveform = taF.resample(waveform, orig_freq=sr, new_freq=WHISPER_SR)

        waveform = whisper.pad_or_trim(waveform)
        mel = whisper.log_mel_spectrogram(waveform, n_mels=self.n_mels).to(waveform.device)

        self._layer_outputs.clear()
        encoder_out = self.encoder(mel)

        embeddings: dict[str, torch.Tensor] = {}
        for layer_name, layer_act in self._layer_outputs.items():
            embeddings[layer_name] = layer_act.flatten(start_dim=1)
        embeddings["ln_post"] = encoder_out.flatten(start_dim=1)
        self._layer_outputs.clear()
        return embeddings


class TransformerAudioEncoder(nn.Module):
    """
    Transformer-based audio encoder, using modules and architecture from OpenAI's Whisper.
    Defaults are set to match OpenAI's Whisper model "tiny" configuration. 
    See: https://github.com/openai/whisper/blob/main/whisper/model.py#L22

    Args:
        n_mels: Number of mel frequency bins
        n_ctx: Number of context frames
        n_state: Number of hidden units
        n_head: Number of attention heads
        n_layer: Number of transformer layers
    """
    def __init__(
        self, n_mels: int=80, n_ctx: int=1500, n_state: int=384, n_head: int=6, n_layer: int=4
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer

        self.conv1 = whisper_models.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = whisper_models.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", whisper_models.sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList([
            whisper_models.ResidualAttentionBlock(n_state, n_head)
            for _ in range(n_layer)
        ])
        
        # TODO: check if this will interact strangelsy with barlow loss
        self.ln_post = whisper_models.LayerNorm(n_state)

    def forward(self, x: torch.Tensor, with_latent: bool = False, fake_relu: bool = False, no_relu: bool = False):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        del fake_relu
        del no_relu
        all_outputs = {} if with_latent else None

        # toss singlechannel dimension if it exists.
        # This is from dataloader for 2d convolutions,
        # which is not needed for 1d convolutions 
        if x.dim() > 3:
            x = x.squeeze(dim=1)
        if with_latent:
            all_outputs["input_after_preproc"] = x
        x = F.gelu(self.conv1(x))
        if with_latent:
            all_outputs["conv1"] = x
        x = F.gelu(self.conv2(x))
        if with_latent:
            all_outputs["conv2"] = x
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, f"incorrect audio shape, got {x.shape[1:]} expected {self.positional_embedding.shape} for n_ctx = {self.n_ctx} and n_state = {self.n_state}"
        x = (x + self.positional_embedding).to(x.dtype)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if with_latent:
                layer_out = x.permute(0, 2, 1)
                all_outputs[f"encoder_block_{idx}"] = layer_out
                all_outputs[f"block_{idx}"] = layer_out

        x = self.ln_post(x)
        if with_latent:
            all_outputs["ln_post"] = x.permute(0, 2, 1)
            all_outputs["final"] = all_outputs["ln_post"]
            rep = x.mean(dim=1)
            return rep, rep, all_outputs
        return x