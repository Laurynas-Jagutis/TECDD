import torch
import torch.nn as nn


class SuTraNNextActivity(nn.Module):
    def __init__(
        self,
        vocab_size_pred: int,
        pad_idx: int,
        max_prefix_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size_pred = vocab_size_pred
        self.pad_idx = pad_idx
        self.max_prefix_len = max_prefix_len

        self.bos_idx = vocab_size_pred
        self.vocab_size_total = vocab_size_pred + 1

        self.tok_emb = nn.Embedding(self.vocab_size_total, d_model, padding_idx=pad_idx)
        self.pos_emb_enc = nn.Embedding(max_prefix_len, d_model)
        self.pos_emb_dec = nn.Embedding(1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.time_proj = nn.Linear(1, d_model)

        self.out_act = nn.Linear(d_model, vocab_size_pred)
        self.out_time = nn.Linear(d_model, 1)

        self.config = {
            "vocab_size_pred": vocab_size_pred,
            "pad_idx": pad_idx,
            "max_prefix_len": max_prefix_len,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        }

    def forward(self, x_prefix: torch.Tensor, x_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, K = x_prefix.shape
        assert K == self.max_prefix_len, f"Expected K={self.max_prefix_len}, got {K}"

        src_key_padding_mask = (x_prefix == self.pad_idx)

        pos_ids_enc = torch.arange(K, device=x_prefix.device).unsqueeze(0).expand(B, K)

        # Each position: activity embedding + linear projection of the time gap + positional embedding
        act_emb = self.tok_emb(x_prefix)
        time_emb = self.time_proj(x_times.unsqueeze(-1))
        enc_in = act_emb + time_emb + self.pos_emb_enc(pos_ids_enc)

        memory = self.encoder(enc_in, src_key_padding_mask=src_key_padding_mask)

        bos = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=x_prefix.device)
        pos_ids_dec = torch.zeros((B, 1), dtype=torch.long, device=x_prefix.device)
        dec_in = self.tok_emb(bos) + self.pos_emb_dec(pos_ids_dec)

        dec_out = self.decoder(
            tgt=dec_in,
            memory=memory,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # Next activity and time until the next activity
        logits_act = self.out_act(dec_out[:, 0, :])
        pred_time = self.out_time(dec_out[:, 0, :])
        return logits_act, pred_time


def save_checkpoint(path, model, extra):
    torch.save({"model_state_dict": model.state_dict(), "model_config": model.config, "extra": extra}, path)


def load_checkpoint(path, device="cpu"):
    """The trained model in eval mode and the full checkpoint dict."""
    ckpt = torch.load(path, map_location=device)
    model = SuTraNNextActivity(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
