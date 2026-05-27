const TAG_PHYS = "phys"
const TAG_HLINK = "hlink"
const TAG_VLINK = "vlink"

phys_tag(row, col) = "$TAG_PHYS,r$(row)c$(col)"
hlink_tag(row, col) = "$TAG_HLINK,r$(row)c$(col)-$(col+1)"
vlink_tag(row, col) = "$TAG_VLINK,r$(row)-$(row+1)c$(col)"

const IndexGrid = Matrix{Index{Int}}

const PepsSkeleton =
    @NamedTuple{tensors::Matrix{ITensor}, phys::IndexGrid, h_links::IndexGrid, v_links::IndexGrid}

struct PepsConfig{T<:BlasFloat}
    bond_dim::Int
    phys_dim::Int
    num_rows::Int
    num_cols::Int

    function PepsConfig{T}(;
        bond_dim::Integer,
        phys_dim::Integer,
        num_rows::Integer,
        num_cols::Integer,
    ) where {T<:BlasFloat}
        num_rows >= 3 || throw(ArgumentError("num_rows must be >= 3, got $num_rows"))
        num_cols >= 3 || throw(ArgumentError("num_cols must be >= 3, got $num_cols"))
        bond_dim >= 1 || throw(ArgumentError("bond_dim must be >= 1, got $bond_dim"))
        phys_dim >= 1 || throw(ArgumentError("phys_dim must be >= 1, got $phys_dim"))
        return new{T}(bond_dim, phys_dim, num_rows, num_cols)
    end
end

struct PEPS{T<:BlasFloat}
    tensors::Matrix{ITensor}
    phys::IndexGrid
    h_links::IndexGrid
    v_links::IndexGrid

    function PEPS{T}(c::PepsConfig{T}) where {T<:BlasFloat}
        skel = build_skeleton(T; c.bond_dim, c.phys_dim, c.num_rows, c.num_cols)
        return new{T}(skel.tensors, skel.phys, skel.h_links, skel.v_links)
    end

    PEPS{T}(
        tensors::Matrix{ITensor},
        phys::IndexGrid,
        h_links::IndexGrid,
        v_links::IndexGrid,
    ) where {T<:BlasFloat} = new{T}(tensors, phys, h_links, v_links)
end

function Base.show(io::IO, p::PEPS{T}) where {T}
    print(
        io,
        "PEPS{$T}(",
        num_rows(p),
        "x",
        num_cols(p),
        ", phys=",
        phys_dim(p),
        ", bond=",
        bond_dim(p),
        ")",
    )
end

bond_dim(peps::PEPS) = dim(peps.h_links[1, 1])
phys_dim(peps::PEPS) = dim(peps.phys[1, 1])
num_rows(peps::PEPS) = size(peps.tensors, 1)
num_cols(peps::PEPS) = size(peps.tensors, 2)
Base.size(peps::PEPS) = size(peps.tensors)
Base.axes(peps::PEPS) = axes(peps.tensors)
Base.getindex(peps::PEPS, inds...) = getindex(peps.tensors, inds...)
Base.firstindex(peps::PEPS, d::Integer) = firstindex(peps.tensors, d)
Base.lastindex(peps::PEPS, d::Integer) = lastindex(peps.tensors, d)

function site_legs(
    phys::IndexGrid,
    h_links::IndexGrid,
    v_links::IndexGrid,
    row::Integer,
    col::Integer,
)
    num_rows, num_cols = size(phys)

    ingoing = Index{Int}[phys[row, col]]
    col > 1 && push!(ingoing, h_links[row, col-1])   # left
    row > 1 && push!(ingoing, v_links[row-1, col])   # up

    outgoing = Index{Int}[]
    col < num_cols && push!(outgoing, h_links[row, col])   # right
    row < num_rows && push!(outgoing, v_links[row, col])   # down

    return ingoing, outgoing
end

function build_skeleton(
    ::Type{T};
    bond_dim::Integer,
    phys_dim::Integer,
    num_rows::Integer,
    num_cols::Integer,
)::PepsSkeleton where {T}
    #! format: off
    phys    = [Index(phys_dim, phys_tag(row, col))
                for row = 1:num_rows, col = 1:num_cols]
    h_links = [Index(bond_dim, hlink_tag(row, col))
                for row = 1:num_rows, col = 1:num_cols-1]
    v_links = [Index(bond_dim, vlink_tag(row, col))
                for row = 1:num_rows-1, col = 1:num_cols]
    #! format: on

    tensors = Matrix{ITensor}(undef, num_rows, num_cols)
    for row = 1:num_rows, col = 1:num_cols
        ingoing, outgoing = site_legs(phys, h_links, v_links, row, col)
        tensors[row, col] = ITensor(T, ingoing..., outgoing...)
    end
    return (; tensors, phys, h_links, v_links)
end

function fill_random_isometry!(peps::PEPS{T}) where {T}
    rows, cols = axes(peps)
    for row in rows, col in cols
        ingoing, outgoing = site_legs(peps.phys, peps.h_links, peps.v_links, row, col)
        m = prod(dim, ingoing)
        n = prod(dim, outgoing; init = 1)
        u = NDTensors.random_unitary(T, m, n)
        peps.tensors[row, col] = ITensor(u, ingoing..., outgoing...)
    end
    return peps
end

random_peps(c::PepsConfig{T}) where {T<:BlasFloat} = fill_random_isometry!(PEPS{T}(c))
