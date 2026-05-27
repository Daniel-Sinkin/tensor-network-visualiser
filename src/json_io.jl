const EXPORT_VERSION = 1

function _index_dict(idx::Index)
    return Dict(
        "id"   => string(ITensors.id(idx)),  # string-encoded because UInt64 ids lose precision in JSON parsers above 2^53
        "dim"  => Int(dim(idx)),
        "tags" => [string(t) for t in ITensors.tags(idx)],
        "plev" => ITensors.plev(idx),
    )
end

function _tensor_data_flat(t::ITensor)
    idxs = collect(inds(t))
    arr  = ITensors.array(t, idxs...)
    flat = vec(arr)
    if eltype(arr) <: Complex
        # interleaved [re0, im0, re1, im1, ...]
        out = Vector{Float64}(undef, 2 * length(flat))
        @inbounds for i in eachindex(flat)
            out[2i - 1] = real(flat[i])
            out[2i]     = imag(flat[i])
        end
        return out
    else
        return [Float64(x) for x in flat]
    end
end

function tensor_envelope(t::ITensor)
    idxs = collect(inds(t))
    return Dict(
        "version" => EXPORT_VERSION,
        "variant" => "Tensor",
        "content" => Dict(
            "eltype"     => string(eltype(t)),
            "indices"    => [_index_dict(i) for i in idxs],
            "data_order" => "col_major",
            "data"       => _tensor_data_flat(t),
        ),
    )
end

function mps_envelope(m::MPS)
    return Dict(
        "version" => EXPORT_VERSION,
        "variant" => "MPS",
        "content" => Dict(
            "length"  => length(m),
            "llim"    => m.llim,
            "rlim"    => m.rlim,
            "tensors" => [tensor_envelope(t) for t in m],
        ),
    )
end

function mpo_envelope(m::MPO)
    return Dict(
        "version" => EXPORT_VERSION,
        "variant" => "MPO",
        "content" => Dict(
            "length"  => length(m),
            "llim"    => m.llim,
            "rlim"    => m.rlim,
            "tensors" => [tensor_envelope(t) for t in m],
        ),
    )
end

function peps_envelope(p::PEPS)
    nr, nc = size(p)
    return Dict(
        "version" => EXPORT_VERSION,
        "variant" => "PEPS",
        "content" => Dict(
            "num_rows" => nr,
            "num_cols" => nc,
            "tensors"  => [[tensor_envelope(p.tensors[r, c]) for c = 1:nc] for r = 1:nr],
            "phys"     => [[_index_dict(p.phys[r, c])    for c = 1:nc]     for r = 1:nr],
            "h_links"  => [[_index_dict(p.h_links[r, c]) for c = 1:nc-1]   for r = 1:nr],
            "v_links"  => [[_index_dict(p.v_links[r, c]) for c = 1:nc]     for r = 1:nr-1],
        ),
    )
end

to_envelope(t::ITensor) = tensor_envelope(t)
to_envelope(m::MPS)     = mps_envelope(m)
to_envelope(m::MPO)     = mpo_envelope(m)
to_envelope(p::PEPS)    = peps_envelope(p)

function export_json(path::AbstractString, x)
    open(path, "w") do io
        JSON3.write(io, to_envelope(x))
    end
    return path
end

function export_jsonl(path::AbstractString, xs)
    open(path, "w") do io
        for x in xs
            JSON3.write(io, to_envelope(x))
            print(io, '\n')
        end
    end
    return path
end

function _tagset_from_strings(tags_vec)
    isempty(tags_vec) && return ITensors.TagSet("")
    return ITensors.TagSet(join(tags_vec, ","))
end

function _parse_eltype(s::AbstractString)
    s == "Float32"    && return Float32
    s == "Float64"    && return Float64
    s == "ComplexF32" && return ComplexF32
    s == "ComplexF64" && return ComplexF64
    error("Unsupported eltype: $s")
end

function _index_from_dict(d)
    return Index(
        parse(ITensors.IDType, String(d["id"])),
        Int(d["dim"]),
        ITensors.Neither,
        _tagset_from_strings(d["tags"]),
        Int(d["plev"]),
    )
end

function _check_envelope(env, expected_variant)
    Int(env["version"]) == EXPORT_VERSION ||
        error("envelope version $(env["version"]) != $EXPORT_VERSION")
    String(env["variant"]) == expected_variant ||
        error("expected variant $expected_variant, got $(env["variant"])")
    return env["content"]
end

function tensor_from_dict(env)
    c = _check_envelope(env, "Tensor")
    String(c["data_order"]) == "col_major" ||
        error("unsupported data_order: $(c["data_order"])")

    T = _parse_eltype(String(c["eltype"]))
    indices = [_index_from_dict(d) for d in c["indices"]]
    dims_tuple = Tuple(Int(d["dim"]) for d in c["indices"])
    raw = c["data"]

    if T <: Complex
        Treal = real(T)
        length(raw) == 2 * prod(dims_tuple) ||
            error("complex data length $(length(raw)) != 2 * prod(dims) = $(2 * prod(dims_tuple))")
        n = length(raw) ÷ 2
        flat = Vector{T}(undef, n)
        @inbounds for i = 1:n
            flat[i] = T(Treal(raw[2i - 1]), Treal(raw[2i]))
        end
    else
        length(raw) == prod(dims_tuple) ||
            error("data length $(length(raw)) != prod(dims) = $(prod(dims_tuple))")
        flat = T[T(x) for x in raw]
    end

    arr = reshape(flat, dims_tuple)
    return ITensor(arr, indices...)
end

function mps_from_dict(env)
    c = _check_envelope(env, "MPS")
    tensors = ITensor[tensor_from_dict(td) for td in c["tensors"]]
    return MPS(tensors, Int(c["llim"]), Int(c["rlim"]))
end

function mpo_from_dict(env)
    c = _check_envelope(env, "MPO")
    tensors = ITensor[tensor_from_dict(td) for td in c["tensors"]]
    return MPO(tensors, Int(c["llim"]), Int(c["rlim"]))
end

function peps_from_dict(env)
    c = _check_envelope(env, "PEPS")
    nr = Int(c["num_rows"])
    nc = Int(c["num_cols"])

    tensors = Matrix{ITensor}(undef, nr, nc)
    for r = 1:nr, col = 1:nc
        tensors[r, col] = tensor_from_dict(c["tensors"][r][col])
    end

    phys    = [_index_from_dict(c["phys"][r][col])    for r = 1:nr,   col = 1:nc]
    h_links = [_index_from_dict(c["h_links"][r][col]) for r = 1:nr,   col = 1:nc-1]
    v_links = [_index_from_dict(c["v_links"][r][col]) for r = 1:nr-1, col = 1:nc]

    T = eltype(tensors[1, 1])
    return PEPS{T}(tensors, phys, h_links, v_links)
end

function from_envelope(env)
    v = String(env["variant"])
    v == "Tensor" && return tensor_from_dict(env)
    v == "MPS"    && return mps_from_dict(env)
    v == "MPO"    && return mpo_from_dict(env)
    v == "PEPS"   && return peps_from_dict(env)
    error("unknown variant: $v")
end

function import_json(path::AbstractString)
    return from_envelope(JSON3.read(read(path, String)))
end

function import_jsonl(path::AbstractString)
    return [from_envelope(JSON3.read(line)) for line in eachline(path)]
end
