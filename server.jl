# Contract server for the tensor-network-visualiser web frontend.
# Launch: julia --project=. server.jl
# Listens on http://127.0.0.1:8754
#
# POST /contract
#   body: { "left": <envelope>, "right": <envelope>, "order": "lr" | "rl" }
#   returns: contracted envelope (MPS or MPO, v1 schema)

using HTTP
using JSON3
using ITensors
using ITensorMPS
using NDTensors

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using TensorNetworkVisualiser
using TensorNetworkVisualiser: from_envelope, mpo_envelope, mps_envelope

const HOST = "127.0.0.1"
const PORT = 8754

cors() = [
    "Content-Type" => "application/json",
    "Access-Control-Allow-Origin" => "*",
    "Access-Control-Allow-Methods" => "POST, OPTIONS",
    "Access-Control-Allow-Headers" => "Content-Type",
]

ok_response(payload)   = HTTP.Response(200, cors(), JSON3.write(payload))
err_response(s, msg)   = HTTP.Response(s,   cors(), JSON3.write(Dict("error" => msg)))

# An Environment payload arrives as a NamedTuple from TensorNetworkVisualiser;
# unwrap to its MPS for contraction.
unwrap_chain(x::MPS) = x
unwrap_chain(x::MPO) = x
unwrap_chain(x::NamedTuple) = haskey(x, :mps) ? x.mps : error("named-tuple operand has no .mps field")
unwrap_chain(x) = error("unsupported operand type: $(typeof(x))")

function chain_mul(A, B)
    NA, NB = length(A), length(B)
    NA == NB || error("chain lengths differ: $NA vs $NB")
    return ITensor[A[k] * B[k] for k in 1:NA]
end

function handle_contract(req::HTTP.Request)
    body = JSON3.read(String(req.body))
    haskey(body, :left)  || return err_response(400, "missing 'left'")
    haskey(body, :right) || return err_response(400, "missing 'right'")
    order = String(get(body, :order, "lr"))
    left  = from_envelope(body[:left])
    right = from_envelope(body[:right])
    A, B = order == "rl" ? (right, left) : (left, right)
    Au = unwrap_chain(A)
    Bu = unwrap_chain(B)
    sites = chain_mul(Au, Bu)
    if Au isa MPO || Bu isa MPO
        return ok_response(mpo_envelope(MPO(sites)))
    else
        return ok_response(mps_envelope(MPS(sites)))
    end
end

function handle(req::HTTP.Request)
    try
        if req.method == "OPTIONS"
            return HTTP.Response(204, cors())
        end
        if req.method == "POST" && req.target == "/contract"
            return handle_contract(req)
        end
        if req.method == "GET" && req.target == "/healthz"
            return ok_response(Dict("ok" => true))
        end
        return err_response(404, "not found: $(req.method) $(req.target)")
    catch err
        @error "request failed" exception=(err, catch_backtrace()) target=req.target
        return err_response(500, sprint(showerror, err))
    end
end

@info "starting contract server" url="http://$HOST:$PORT"
HTTP.serve(handle, HOST, PORT)
