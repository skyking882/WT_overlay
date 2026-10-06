import unittest

import common  # noqa: F401
import torch
import torch.nn as nn

from rl import spec
from rl.model import Actor, Critic, EntityEncoder, PreLNBlock, count_params, gru_unroll


class ParamCounts(unittest.TestCase):
    def test_totals_match_spec(self):
        # spec 13.2: the antenna head has 5 options -> actor 643,125 (the section-6 table's 642,355 + 2 x 385)
        self.assertEqual(count_params(Actor()), 643_125)
        self.assertEqual(count_params(Critic()), 650_627)

    def test_original_3_option_antenna_gives_the_section_6_table_total(self):
        from unittest import mock
        with mock.patch.dict(spec.CAT_SIZES, {"antenna": 3}):
            self.assertEqual(count_params(Actor()), 642_355)
            self.assertEqual(count_params(Actor().cat), 16_050)

    def test_breakdown_matches_spec_table(self):
        a, c = Actor(), Critic()
        n = count_params
        self.assertEqual(n(a.ent), 74_305)
        self.assertEqual(n(a.own), 28_928)
        self.assertEqual(n(a.fuse) + n(a.fuse_ln), 62_208)
        self.assertEqual(n(a.gru), 394_752)
        self.assertEqual(n(a.ptr), 66_112)
        self.assertEqual(n(a.cat), 16_050 + 2 * 385)             # 13.2: antenna 3 -> 5 options
        self.assertEqual(n(c.ent), 74_305)
        self.assertEqual(n(c.own), 28_928)
        self.assertEqual(n(c.truth), 73_793)
        self.assertEqual(n(c.fuse) + n(c.fuse_ln), 78_592)
        self.assertEqual(n(c.gru), 394_752)
        self.assertEqual(n(c.value), 257)
        self.assertEqual(n(a) + n(c), 1_292_982 + 770)

    def test_pointer_head_inputs(self):
        a = Actor()
        self.assertEqual(a.ptr["maneuver_ref"].q.in_features, 256)
        self.assertEqual(a.ptr["target"].q.in_features, 256 + 64)      # + selected maneuver reference
        self.assertEqual(a.ptr["view_object"].q.in_features, 256)
        for h in ("maneuver", "vertical", "speed", "chaff"):
            self.assertEqual(a.cat[h].in_features, 256 + 64)
        for h in ("radar_mode", "antenna", "weapon", "kb_roll", "kb_pitch"):
            self.assertEqual(a.cat[h].in_features, 256 + 128)
        for h in ("view_mode", "look_az", "look_el"):
            self.assertEqual(a.cat[h].in_features, 256)
        self.assertEqual(sum(spec.CAT_SIZES.values()), 52)
        self.assertEqual(a.cat["antenna"].out_features, 5)
        self.assertEqual(len(a.cat), 12)


class Structure(unittest.TestCase):
    def test_block_equals_torch_pre_ln_encoder_layer(self):
        torch.manual_seed(0)
        blk = PreLNBlock()
        ref = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.0,
                                         activation="gelu", batch_first=True, norm_first=True)
        with torch.no_grad():
            ref.self_attn.in_proj_weight.copy_(blk.qkv.weight)
            ref.self_attn.in_proj_bias.copy_(blk.qkv.bias)
            ref.self_attn.out_proj.weight.copy_(blk.proj.weight)
            ref.self_attn.out_proj.bias.copy_(blk.proj.bias)
            ref.linear1.weight.copy_(blk.ff1.weight); ref.linear1.bias.copy_(blk.ff1.bias)
            ref.linear2.weight.copy_(blk.ff2.weight); ref.linear2.bias.copy_(blk.ff2.bias)
            ref.norm1.weight.copy_(blk.ln1.weight); ref.norm1.bias.copy_(blk.ln1.bias)
            ref.norm2.weight.copy_(blk.ln2.weight); ref.norm2.bias.copy_(blk.ln2.bias)
        ref.train()
        x = torch.randn(5, 9, 64)
        valid = torch.rand(5, 9) > 0.4
        valid[:, 0] = True
        with torch.no_grad():
            a = blk(x, valid)
            b = ref(x, src_key_padding_mask=~valid)
        self.assertLess((a - b).abs().max().item(), 1e-5)

    def test_encoder_has_no_positional_encoding_and_is_permutation_invariant(self):
        torch.manual_seed(1)
        enc = EntityEncoder(48).eval()
        x = torch.randn(3, 7, 48)
        m = torch.ones(3, 7, dtype=torch.bool)
        perm = torch.randperm(7)
        with torch.no_grad():
            h1, p1 = enc(x, m)
            h2, p2 = enc(x[:, perm], m)
        self.assertLess((p1 - p2).abs().max().item(), 1e-5)
        self.assertLess((h1[:, perm] - h2).abs().max().item(), 1e-5)

    def test_empty_entity_set_gives_zero_pool_and_no_nan(self):
        enc = EntityEncoder(48)
        h, p = enc(torch.randn(2, 4, 48), torch.zeros(2, 4, dtype=torch.bool))
        self.assertTrue(torch.isfinite(h).all() and torch.isfinite(p).all())
        self.assertEqual(p.abs().max().item(), 0.0)

    def test_gru_unroll_matches_nn_gru_without_resets(self):
        torch.manual_seed(2)
        cell = nn.GRUCell(256, 256)
        gru = nn.GRU(256, 256, batch_first=True)
        with torch.no_grad():
            gru.weight_ih_l0.copy_(cell.weight_ih); gru.weight_hh_l0.copy_(cell.weight_hh)
            gru.bias_ih_l0.copy_(cell.bias_ih); gru.bias_hh_l0.copy_(cell.bias_hh)
        x = torch.randn(3, 11, 256)
        h0 = torch.randn(3, 256)
        T = torch.ones(3, 11, dtype=torch.bool)
        F_ = torch.zeros(3, 11, dtype=torch.bool)
        out, hl = gru_unroll(cell, x, h0, F_, T)
        ref, hr = gru(x, h0.unsqueeze(0))
        self.assertLess((out - ref).abs().max().item(), 1e-5)
        self.assertLess((hl - hr[0]).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
