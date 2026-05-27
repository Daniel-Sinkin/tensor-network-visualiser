module TensorNetworkVisualiser

import ITensorMPS
import ITensors
import NDTensors

using ITensorMPS: MPO, MPS
using ITensors: Index, ITensor, dim, inds
using LinearAlgebra: BlasFloat
using dans_tn_core

include("peps.jl")

# The simulator emits Environment payloads inside Snapshot frames. The
# visualiser has no reason to depend on the simulator's struct hierarchy, so
# parse them into a lightweight NamedTuple. .mps is reconstructed as a real
# MPS via the builtin dans_tn_core parser.
function environment_from_dict(env)
    Int(env["version"]) == EXPORT_VERSION || error("envelope version mismatch")
    String(env["variant"]) == "Environment" || error("expected variant Environment")
    c = env["content"]
    return (
        layer   = String(c["layer"]),
        lognorm = Float64(c["lognorm"]),
        mps     = from_envelope(c["mps"]),
    )
end

function __init__()
    register_variant!("Environment", environment_from_dict)
end

export PEPS, PepsConfig, random_peps, peps_from_grid
export EXPORT_VERSION
export to_envelope, from_envelope
export tensor_envelope, mps_envelope, mpo_envelope, tensor_grid_envelope
export tensor_from_dict, mps_from_dict, mpo_from_dict, tensor_grid_from_dict
export export_json, export_jsonl, import_json, import_jsonl
export SnapshotWriter, trace_to, snapshot!, current_snapshot_writer
export register_variant!

end # module
