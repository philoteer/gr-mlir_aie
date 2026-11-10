#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parameter validation must fail before opening files or an XRT device."""
from gnuradio import gr, gr_unittest, mlir_aie


class qa_mlir_aie_80211_phy_thru_mac_decode(gr_unittest.TestCase):
    def test_invalid_parameters(self):
        for vector, frequency, slots, tiles in (
            (0, 2412e6, 2, 16),
            (16384, 2412e6, 0, 16),
            (16384, 2412e6, 2, 0),
            (16383, 2412e6, 2, 16),
            (8192, 2412e6, 2, 16),
            (16384, -1, 2, 16),
            (16384, float('nan'), 2, 16),
        ):
            with self.subTest(vector=vector, frequency=frequency, slots=slots, tiles=tiles):
                with self.assertRaises(ValueError):
                    mlir_aie.mlir_aie_80211_phy_thru_mac_decode(
                        'missing.xclbin', 'missing.bin', 'MLIR_AIE',
                        vector, frequency, slots, tiles)


if __name__ == '__main__':
    gr_unittest.run(qa_mlir_aie_80211_phy_thru_mac_decode)
