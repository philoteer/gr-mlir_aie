/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_80211_PHY_EVM_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_80211_PHY_EVM_H

#include <gnuradio/block.h>
#include <gnuradio/mlir_aie/api.h>

namespace gr {
namespace mlir_aie {

/*!
 * \brief Run an 802.11 PHY MLIR-AIE kernel.
 * \ingroup mlir_aie
 *
 */
class MLIR_AIE_API mlir_aie_80211_phy_evm : virtual public gr::block
{
public:
    typedef std::shared_ptr<mlir_aie_80211_phy_evm> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of mlir_aie::mlir_aie_80211_phy_evm.
     *
     * To avoid accidental use of raw pointers, mlir_aie::mlir_aie_80211_phy_evm's
     * constructor is in a private implementation
     * class. mlir_aie::mlir_aie_80211_phy_evm::make is the public interface for
     * creating new instances.
     */
    static sptr make(const char* path_xclbin,
                     const char* path_insts_bin,
                     const char* kernel_name,
                     int VECTOR_SIZE,
                     double nominal_frequency,
                     int num_slots = 2,
                     int N_TILES = 4,
                     const char* weights_path = "");

    virtual void set_nominal_frequency(double nominal_frequency) = 0;
    virtual double nominal_frequency() const = 0;
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_MLIR_AIE_80211_PHY_EVM_H */
