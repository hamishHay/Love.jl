# Julia code to calculate tidal deformation, Love numbers, and heating
# This version is based on the method outlined in Kamata (2023).
# Author: H. Hay

# μ: solid shear modulus
# ρ: solid density 
# ρₗ: liquid density 
# K: solid bulk modulus 
# Kl: liquid bulk modulus 
# α: Biot's constant 
# λ: Lame's First Parameter
# η: solid viscosity 
# ηₗ: liquid viscosity 
# g: gravity 
# ϕ: porosity 
# k: permeability 
# ω: rotation rate


module Love

    using LinearAlgebra
    using DoubleFloats
    using AssociatedLegendrePolynomials
    include("SphericalHarmonics.jl")
    using .SphericalHarmonics
    using StaticArrays

    export get_g, get_A!, get_A, get_B_product!, get_Ic, get_B, get_B!
    export expand_layers, set_G, compute_y
    export get_displacement, get_darcy_velocity, get_solution, get_solution_old
    export get_total_heating, get_heating_profile, get_heating_map
    export define_spherical_grid
    export get_radial_isotropic_coeffs
    export get_ke_power

    # Depending on the stability of the system, different precision can be 
    # chosen here. Low mobility is a challenge, so by default I have set 
    # the precision to 128 bits. 

    # prec = Float64 #BigFloat
    # precc = ComplexF64 #Complex{BigFloat}

    # prec = Double64 #BigFloat
    # precc = ComplexDF64 #Complex{BigFloat}

    prec = BigFloat
    precc = Complex{BigFloat}

    G = prec(6.6743e-11)
    n = 2
    nr = 100

    porous = false

    M = 6 + 2porous         # Matrix size: 6x6 if only solid material, 8x8 for two-phases

    Abot_p = zeros(precc, 8, 8)
    Amid_p = zeros(precc, 8, 8)
    Atop_p = zeros(precc, 8, 8)

    k18 = zeros(precc, 8, 8)
    k28 = zeros(precc, 8, 8)
    k38 = zeros(precc, 8, 8)
    k48 = zeros(precc, 8, 8)

    k16 = zeros(precc, 6, 6)
    k26 = zeros(precc, 6, 6)
    k36 = zeros(precc, 6, 6)
    k46 = zeros(precc, 6, 6)

    # I8 = Matrix{Float64}(I, 8, 8)
    I8 = SMatrix{8,8,precc}(I)
    I6 = SMatrix{6,6,precc}(I)

    Abot = zeros(precc, 6, 6)
    Amid = zeros(precc, 6, 6)
    Atop = zeros(precc, 6, 6)

    clats = 0.0
    lons = 0.0
    Y = 0.0
    dYdθ = 0.0
    dYdϕ = 0.0
    Z = 0.0
    X = 0.0
    res = 0.0

    """
        set_G(new_G)

    Overwrite the value of the Universal Gravitational Constant. Only to be used for non-dimensional calculations!
    """
    function set_G(new_G)
        Love.G = new_G
    end

    function set_nr(new_nr)
        Love.nr = new_nr
    end

    """
        get_g(r, ρ)

    Compute the radial gravity structure associated with a density profile `r` at intervals given by `r`.

    `r` must be be a 2D array, with index 1 representing the top radius of secondary layers, and index 2
    representing the top radius of primary layers. 
    """
    function get_g(r, ρ)
        g = zeros(Float64, size(r))
        M = zeros(Float64, size(r))

        for i in 1:size(r)[2]
            M[2:end,i] = 4.0/3.0 * π .* diff(r[:,i].^3) .* ρ[i]
        end
    
        g[2:end,:] .= G*accumulate(+,M[2:end,:]) ./ r[2:end,:].^2
        g[1,2:end] = g[end,1:end-1]

        return g
    end

    """
        get_A(r, ρ, g, μ, K)

    Compute the 6x6 `A` matrix in the ODE for the solid-body problem.

    See also [`get_A!`](@ref)
    """
    function get_A(r, ρ, g, μ, K)
        A = zeros(precc, 6, 6) 
        get_A!(A, r, ρ, g, μ, K)
        return A
    end

    """
        get_A!(r, ρ, g, μ, K; λ=nothing)

    Compute the 6x6 `A` matrix in the ODE for the solid-body problem. These correspond to 
    the coefficients given in Equation S4.6 in Hay et al., (2025) when α=φ=0, as well as Sabadini and Vermeersen 
    (2016) Eq. 1.95.

    See also [`get_A`](@ref)
    """
    function get_A!(A::Matrix, r, ρ, g, μ, K; λ=nothing)
        if isnothing(λ)
            λ = K - 2μ/3
        end

        r_inv = 1.0/r
        β_inv = 1.0/(2μ + λ)
        rβ_inv = r_inv * β_inv

        A[1,1] = -2λ * r_inv*β_inv
        A[2,1] = -r_inv
        A[3,1] = 4r_inv * (3K*μ*r_inv*β_inv - ρ*g)       #- ω^2 * ρ# 
        A[4,1] = -r_inv * (6K*μ*r_inv*β_inv - ρ*g )
        A[5,1] = 4π * G * ρ
        A[6,1] = 4π*(n+1)*G*ρ*r_inv

        A[1,2] = n*(n+1) * λ * r_inv*β_inv
        A[2,2] = r_inv
        A[3,2] = -n*(n+1)*r_inv * (6K*μ*r_inv*β_inv - ρ*g ) 
        A[4,2] = 2μ*r_inv^2 * (n*(n+1)*(1 + λ*β_inv) - 1.0 ) #- ω^2 * ρ# 
        A[6,2] = -4π*n*(n+1)*G*ρ*r_inv

        A[1,3] = β_inv
        A[3,3] = r_inv*β_inv * (-4μ )
        A[4,3] = -λ * r_inv*β_inv
        
        A[2,4] = 1.0 / μ
        A[3,4] = n*(n+1)*r_inv
        A[4,4] = -3r_inv

        A[3,5] = ρ * (n+1)*r_inv
        A[4,5] = -ρ*r_inv
        A[5,5] = -(n+1)r_inv     

        A[3,6] = -ρ
        A[5,6] = 1.0
        A[6,6] = (n-1)r_inv
    end

    """
        get_A(A::Matrix, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

    Compute the 8x8 `A` matrix in the ODE for the two-phase problem. These correspond to 
    the coefficients given in Equation S4.6 in Hay et al., (2025).

    See also [`get_A!`](@ref)
    """
    function get_A(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        A = zeros(precc, 8, 8)
        get_A!(A, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        return A
    end

    """
        get_A!(A::Matrix, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

    Compute the 8x8 `A` matrix in the ODE for the two-phase problem. These correspond to 
    the coefficients given in Equation S4.6 in Hay et al., (2025).

    See also [`get_A`](@ref)
    """
    function get_A!(A::Matrix, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        λ = Kd .- 2μ/3       # Lame's second param, which uses the drained compaction modulus
        S = ϕ/Kl + (α - ϕ)/K # Storavity, which uses liquid and solid grain bulk moduli  

        # First add the solid-body coefficients, but using drained moduli. 
        get_A!(A, r, ρ, g, μ, Kd; λ=λ)    # Note that here we replace the bulk modulus with the compaction modulus

        r_inv = 1.0/r
        β_inv = 1.0/(2μ + λ)

        # If there is a porous layer, now add the two-phase components
        if !iszero(ϕ)
            A[1,7] = α * β_inv

            A[3,1] += 1im * k*ρₗ^2 *g^2 * n*(n+1) / (ω*ηₗ) * r_inv^2
            A[3,5] += -(n+1)r_inv * 1im *(k*ρₗ^2*g*n)/(ω*ηₗ) * r_inv                               
            A[3,7] = 1im * (k*ρₗ*g*n*(n+1))/(ω*ηₗ)*r_inv^2 - 4μ*α*β_inv*r_inv 
            A[3,8] =  1im * (k*ρₗ^2*g^2*n*(n+1))/(ω*ηₗ)*r_inv^2 - 4ϕ*ρₗ*g*r_inv 
        
            A[4,7] = 2α*μ*r_inv * β_inv
            A[4,8] = ϕ*ρₗ*g*r_inv 
            
            A[5,8] = 4π*G*ρₗ*ϕ

            A[6,1] += -1im * 4π*G*n*(n+1)*r_inv * (k*ρₗ^2*g/(ω*ηₗ)*r_inv)
            A[6,5] = 1im*4π*n*(n+1)G*(ρₗ)^2*k*r_inv^2 / (ω*ηₗ)  
            A[6,7] = -1im *4π*n*(n+1)G*ρₗ*k*r_inv^2 / ( ω*ηₗ) 
            A[6,8] = 4π*G*(n+1)*r_inv * (ϕ*ρₗ - 1im * n*k*ρₗ^2*g/(ω*ηₗ)*r_inv) 
            
            A[7,1] = ρₗ*g*r_inv * ( 4 - 1im *(k*ρₗ*g*n*(n+1)/(ω*ϕ*ηₗ))*r_inv)  
            A[7,2] = -ρₗ*n*(n+1)*r_inv*g
            A[7,5] = -ρₗ*(n+1)r_inv * (1 - 1im*(k*ρₗ*g*n)/(ω*ϕ*ηₗ)*r_inv)  
            A[7,6] = ρₗ 
            A[7,7] = - 1im*(k*ρₗ*g*n*(n+1))/(ω*ϕ*ηₗ)*r_inv^2
            A[7,8] = -1im*ω*ϕ*ηₗ/k - 4π*G*(ρ - ϕ*ρₗ)*ρₗ + ρₗ*g*r_inv*(4 - 1im*(k*ρₗ*g*n*(n+1))/(ω*ϕ*ηₗ)*r_inv) 
        
            A[8,1] = r_inv*( 1im * k*ρₗ*g*n*(n+1)/(ω*ϕ*ηₗ)*r_inv - α/ϕ * 4μ*β_inv) 
            A[8,2] = α/ϕ * 2n*(n+1)*μ *β_inv * r_inv
            A[8,3] = -α/ϕ * β_inv 
            A[8,5] = -1im * k *ρₗ *n*(n+1) / (ω*ϕ*ηₗ)*r_inv^2 
            A[8,7] = 1im*k*n*(n+1)/(ω*ϕ*ηₗ)*r_inv^2 - 1/ϕ * (S + α^2 * β_inv) # If solid and liquid are compressible, keep the 1/M term
            A[8,8] = 1im * k *ρₗ*g *n*(n+1) / (ω*ϕ*ηₗ)*r_inv^2  - 2r_inv 
        end
        
    end

    "See ['get_B!'](@ref) for definition."
    function get_B(r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        B = zeros(precc, 8, 8)
        get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

        return B
    end

    """
        get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

    Compute the 8x8 numerical integrator matrix, which integrates dy/dr from `r1` to `r2` for the two-phase problem.

    `B` here represnts the RK4 integrator, given by Eq. S5.5 in Hay et al., (2025).

    See also [`get_B`](@ref)
    """
    function get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        dr = r2 - r1
        rhalf = r1 + 0.5dr
        
        ghalf = g1 + 0.5*(g2 - g1)

        get_A!(Abot_p, r1, ρ, g1, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        get_A!(Amid_p, rhalf, ρ, ghalf, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        get_A!(Atop_p, r2, ρ, g2, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        
        k18 .= dr * Abot_p 
        k28 .= dr *  (Amid_p .+ 0.5Amid_p *k18) 
        k38 .= dr *  (Amid_p .+ 0.5*Amid_p *k28)
        k48 .= dr *  (Atop_p .+ Atop_p*k38) 

        B .= (I8 + 1.0/6.0 .* (k18 .+ 2*(k28 .+ k38) .+ k48))
    end

    "See ['get_B!'](@ref) for definition."
    function get_B(r1, r2, g1, g2, ρ, μ, K)
        B = zeros(precc, 6, 6)
        get_B!(B, r1, r2, g1, g2, ρ, μ, K)
        return B
    end

    """
        get_B!(B, r1, r2, g1, g2, ρ, μ, K)

    Compute the 6x6 numerical integrator matrix, which integrates dy/dr from `r1` to `r2` for the solid-body problem.

    `B` here represnts the RK4 integrator, given by Eq. S5.5 in Hay et al., (2025).

    See also [`get_B`](@ref)
    """
    function get_B!(B, r1, r2, g1, g2, ρ, μ, K)
        dr = r2 - r1
        rhalf = r1 + 0.5dr
        
        ghalf = g1 + 0.5*(g2 - g1)

        A1 = get_A(r1, ρ, g1, μ, K)
        Ahalf = get_A(rhalf, ρ, ghalf, μ, K)
        A2 = get_A(r2, ρ, g2, μ, K)
        
        k16 .= dr * A1 
        k26 .= dr * Ahalf * (I + 0.5k16)
        k36 .= dr * Ahalf * (I + 0.5k26)
        k46 .= dr * A2 * (I + k36) 

        # Only compute over the first six indices
        B[1:6,1:6] .= (I + 1.0/6.0 * (k16 + 2k26 + 2k36 + k46))
    end

    """
        get_B_product!(Brod, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

    Compute the product of the 8x8 B matrices within a primary layer. This is used to propgate the
    y solution across a single two-phase primary layer.

    Bprod is denoted by D in Eq. S5.14 in Hay et al., (2025).
    """
    function get_B_product!(Bprod2, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        # Check dimensions of Bprod2

        nr = size(r)[1]

        Bstart = zeros(precc, 8, 8)
        B = zeros(precc, 8, 8)

        for i in 1:6
            Bstart[i,i,1] = 1
        end

        # if layer is porous, 
        # don't filter out y7 and y8 components
        if ϕ>0
            Bstart[7,7,1] = 1
            Bstart[8,8,1] = 1  
        end

        r1 = r[1]
        g1 = g[1]
        for j in 1:nr-1
            r2 = r[j+1]
            g2 = g[j+1]

            if ϕ>0 
                get_B!(B, r1, r2, g1, g2, ρ, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
            else
                get_B!(B, r1, r2, g1, g2, ρ, μ, K)
            end

            Bprod2[:,:,j] .= B * (j==1 ? Bstart : @view(Bprod2[:,:,j-1])) 

            r1 = r2
            g1 = g2 
        end
    end

    """
        get_B_product!(Brod, r, ρ, g, μ, K)

    Compute the product of the 6x6 B matrices within a primary layer. This is used to propgate the
    y solution across one single-phase (solid) primary layer.

    Bprod is denoted by D in Eq. S5.14 in Hay et al., (2025).
    """
    function get_B_product!(Bprod2, r, ρ, g, μ, K)
        Bstart = zeros(precc, 6, 6)
        B = zeros(precc, 6, 6)

        for i in 1:6
            Bstart[i,i,1] = 1.0
        end

        nr = size(r)[1]

        r1 = r[1]
        for j in 1:nr-1
            r2 = r[j+1]
            g1 = g[j]
            g2 = g[j+1]

            get_B!(B, r1, r2, g1, g2, ρ, μ, K)
            Bprod2[:,:,j] .= B * (j==1 ? Bstart : Bprod2[:,:,j-1])

            r1 = r2
        end
    end

    """
        get_solution(y, n, m, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

    Expand the tidal solution from y at degree `n` and order `m`. 

    # Prerequisite calls 

    Note that [`define_spherical_grid()`](@ref) must be called first.
    
    # Returns 
    
    A tuple of the complex fourier coefficients on a lat-lon grid for 
     1. displacement, u 
     2. solid strain, εs 
     3. total stress, σ 
     4. pore pressure, pl 
     5. darcy (relative) displacement, u_rel
     
    For tensors, the array dimensions represent (nlat, nlon, 6xtensor components, secondary lay., primary lay.)
    For vectors, the array dimensions represent (nlat, nlon, 3xvector components, secondary lay., primary lay.)
    For scalars, the array dimensions represent (nlat, nlon, secondary lay., primary lay.)
    
    To recover the real solution at time t, the forcing magnitude and direction
    must be known. 

    # Example

        sol_22 = get_solution(y1, 2, 2, rr, ρ, g, μ, κs, ω, ρl, κl, κd, α, ηl, ϕ, k);
        sol_20  = get_solution(y1, 2,  0, rr, ρ, g, μ, κs, ω, ρl, κl, κd, α, ηl, ϕ, k);

        # eccentricity forcing 
        U22E =  7/8 * ω^2*R^2*ecc 
        U22W = -1/8 * ω^2*R^2*ecc 
        U20  = -3/2 * ω^2*R^2*ecc 

        Y = Love.clats;
        X = Love.lons;
        P20 = 0.5 .* (3 .* cos.(Y).^2 .- 1)
        P22 = 3 .* (1. .- cos.(Y).^2)

        # define phase of solution (0 to 2pi)
        ωt = 0.0

        # Obtain the radial tide
        # Combine the solutions - note that the west component is a complex conjugate
        tide = (U22E*sol_22[1] + U20*sol_20[1])[:,:,1,end,end] 
                .+ U22W*conj.(sol_22[1])[:,:,1,end,end];
        tide = 0.5real.(tide .* exp(-1im * ωt) .+ conj.(tide) .* exp(1im * ωt));
    
    """
    function get_solution(y, n, m, r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)
        R = r[end,end]

        disp = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])
        ϵ = zeros(ComplexF64, length(clats), length(lons), 6, size(r)[1]-1, size(r)[2])
        p = zeros(ComplexF64, length(clats), length(lons), size(r)[1]-1, size(r)[2])
        σ = zero(ϵ)
        d_disp = zero(disp)
    
        if m == -2
            i=1
        elseif m == 0
            i=2
        elseif m == 2
            i=3
        else
            error("m must be -2, 0, or 2")
        end

        for i in 2:size(r)[2] # Loop over primary layers
            ηlr = ηₗ[i]
            ρlr = ρₗ[i]
            ρr = ρ[i]
            kr  = k[i]
            Klr = Kl[i]
            Kr = K[i]
            μr = μ[i]
            ϕr = ϕ[i]
            Kdr = Kd[i]
            αr = α[i]

            for j in 1:size(r)[1]-1 # Loop over sublayers 
                yrr = ComplexF64.(y[:,j,i])

                rr = r[j,i]
                gr = g[j,i]
                
                if ϕ[i] > 0 
                    compute_darcy_displacement!(@view(d_disp[:,:,:,j,i]), yrr, m, rr, ω, ϕr, ηlr, kr, gr, ρlr)
                    compute_pore_pressure!(@view(p[:,:,j,i]), yrr, m)
                end

                compute_displacement!(@view(disp[:,:,:,j,i]), yrr, m)
                compute_strain_ten!(@view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, Kr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
                compute_stress_ten!(@view(σ[:,:,:,j,i]), @view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, Kr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
            end
        end

        return disp, ϵ, σ, p, d_disp
    end

    """
        get_solution(y, n, m, r, ρ, g, μ, K, ω)

    Expand the tidal solution from y at degree `n` and order `m`. 

    # Prerequisite calls 

    Note that [`define_spherical_grid()`](@ref) must be called first.
    
    # Returns 
    
    A tuple of the complex fourier coefficients on a lat-lon grid for 
     1. displacement, u 
     2. solid strain, εs 
     3. total stress, σ 
     
    For tensors, the array dimensions represent (nlat, nlon, 6xtensor components, secondary lay., primary lay.)
    For vectors, the array dimensions represent (nlat, nlon, 3xvector components, secondary lay., primary lay.)
    For scalars, the array dimensions represent (nlat, nlon, secondary lay., primary lay.)
    """
    function get_solution(y, n, m, r, ρ, g, μ, K, ω, ecc)
        R = r[end,end]

        disp = zeros(ComplexF64, length(clats), length(lons), 3, size(r)[1]-1, size(r)[2])
        ϵ = zeros(ComplexF64, length(clats), length(lons), 6, size(r)[1]-1, size(r)[2])
        σ = zero(ϵ)

        n = 2
        ms = [-2, 0, 2]
        forcing = [-1/8, -3/2, 7/8] * ω^2*R^2*ecc 

        if m == -2
            i=1
        elseif m == 0
            i=2
        elseif m == 2
            i=3
        else
            error("m must be -2, 0, or 2")
        end

        for i in 2:size(r)[2] # Loop of secondary layers
            ρr = ρ[i]
            Kr = K[i]
            μr = μ[i]

            for j in 1:size(r)[1]-1 # Loop over sublayers 
                yrr = ComplexF64.(y[:,j,i])

                rr = r[j,i]
                gr = g[j,i]

                compute_displacement!(@view(disp[:,:,:,j,i]), yrr, m)
                compute_strain_ten!(@view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, Kr)
                compute_stress_ten!(@view(σ[:,:,:,j,i]), @view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, Kr)

            end
        end

        return disp, ϵ, σ
    end

    "Convert input parameters into the required precision."
    function convert_params_to_prec(r, ρ, g, μ, κs, ω, ρl, κl, κd, α, ηl, ϕ, k)
        r_prec = convert(Array{prec}, r)
        ρ_prec = convert(Array{prec}, ρ)
        g_prec = convert(Array{prec}, g)
        μ_prec = convert(Array{precc}, μ)
        κs_prec = convert(Array{precc}, κs)
        ρl_prec = convert(Array{prec}, ρl)
        κl_prec = convert(Array{prec}, κl)
        κd_prec = convert(Array{precc}, κd)
        α_prec = convert(Array{precc}, α)
        ηl_prec = convert(Array{prec}, ηl)
        ϕ_prec = convert(Array{prec}, ϕ)
        k_prec = convert(Array{prec}, k)
        ω_prec = convert(prec, ω)

        return (r_prec,  ρ_prec, g_prec, μ_prec, κs_prec, ω_prec, ρl_prec, κl_prec, κd_prec, α_prec, ηl_prec, ϕ_prec, k_prec)
    end

    "Convert input parameters into the required precision."
    function convert_params_to_prec(r, ρ, g, μ, κs)
        r_prec = convert(Array{prec}, r)
        ρ_prec = convert(Array{prec}, ρ)
        g_prec = convert(Array{prec}, g)
        μ_prec = convert(Array{precc}, μ)
        κs_prec = convert(Array{precc}, κs)

        return (r_prec,  ρ_prec, g_prec, μ_prec, κs_prec)
    end

    """
        compute_y(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k, core="liquid")

    Compute the two-phase tidal solution y with the propagator matrix method. 
    
    Returns `y`, an array with size (8, `nr`, `nlay`), where `nr` is the
    number of secondary layers, and `nlay` is the number of primary layers, 
    e.g., core, mantle, crust. The first index corresponds to y1 to y8, where

    y1 = radial displacement \\
    y2 = tangential displacement \\
    y3 = radial stress \\
    y4 = tangential stress \\
    y5 = gravitational potential \\
    y6 = potential stress \\
    y7 = pore pressure \\
    y8 = radial Darcy displacement \\

    # Example
        # Define four layer body
        ρs = [7640, 3300, 3300, 3000.]  # Solid density
        μ =  [60, 60, 60, 60] .* 1e9    # Shear modulus
        κs = [100, 200, 200, 200].*1e9  # Solid bulk modulus
        η =  [1e25, 1e25, 1e14, 1e25]   # Shear viscosity
        ζ =  [1e25, 1e25, 1e15, 1e25]   # Compaction viscosity

        ρl = [0, 0, 3300, 0]            # Liquid density
        κl = [0, 0, 1e9, 0]             # Liquid bulk modulus
        κd = 0.01κs                     # Drained compaction modulus
        k =  [0, 0, 1e-7, 0]            # Permeability
        α =  1.0 .- κd ./ κs            # Biot's modulus
        ηl = [0, 0, 1.0, 0]             # Liquid viscosity
        ϕ =  [0, 0, 0.1, 0]             # Porosity

        ρ = (1 .- ϕ) .* ρs + ϕ .* ρl    # Bulk density
 
        # Discretize primary layers into 100 secondary layers
        rr = expand_layers(r, nr=100)   
        g = get_g(rr, ρ);               # get gravity profile 

        y1 = compute_y(rr, ρ, g, μ, κs, ω, ρl, κl, κd, α, ηl, ϕ, k);
    """
    function compute_y(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k; core="liquid")
        porous_layer = ϕ .> 0.0

        ## Convert parameters to the precision of precc:
        r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k = convert_params_to_prec(r, ρ, g, μ, K, ω, ρₗ, Kl, Kd, α, ηₗ, ϕ, k)

        sum(porous_layer) > 1.0 && error("Can only handle one porous layer for now!")

        nlayers = size(r)[2]
        nsublayers = size(r)[1]

        # Define starting vector as the core solution matrix, Y_r_C (Eq. S5.15)
        y_start = get_Ic(r[end,1], ρ[1], g[end,1], μ[1], core; M=8, N=4)

        y1_4 = zeros(precc, 8,   4, nsublayers-1, nlayers)  # Four linearly independent y solutions
        y =    zeros(ComplexF64, 8, nsublayers-1, nlayers)  # Final y solutions to return, lower precision fine
        
        for i in 2:nlayers
            Bprod = zeros(precc, 8, 8, nsublayers-1) # D matrix from Eq. S5.13
            @views get_B_product!(Bprod, r[:,i], ρ[i], g[:,i], μ[i], K[i], ω, ρₗ[i], Kl[i], Kd[i], α[i], ηₗ[i], ϕ[i], k[i])

            # Modify starting vector if the layer is porous
            # If a new porous layer (i.e., sitting on a non-porous layer)
            # reset the pore pressure and darcy flux
            if porous_layer[i] && !porous_layer[i-1]
                y_start[7,4] = 1.0          # Non-zero pore pressure
                y_start[8,4] = 0.0          # Zero radial Darcy flux
            elseif !porous_layer[i]
                y_start[7:8, :] .= 0.0      # Pore pressure and flux undefined
            end

            for j in 1:nsublayers-1
                y1_4[:,:,j,i] = @view(Bprod[:,:,j]) * y_start 
            end

            y_start[:,:] .= @view(y1_4[:,:,end,i])
        end

        # Get solution at the surface and porous layer interface (Eq. S5.20)
        M = zeros(precc, 4,4)
        
        M[1, :] .= y1_4[3,:,end,end] # Row 1 - Radial Stress
        M[2, :] .= y1_4[4,:,end,end] # Row 2 - Tangential Stress
        M[3, :] .= y1_4[6,:,end,end] # Row 3 - Potential Stress
        
        for i in 2:nlayers
            if porous_layer[i]
                M[4, :] .= y1_4[8,:,end,i]  #  Row 4 - Darcy flux (r = r_tp)
            end
        end

        # Boundary condition vector
        b = zeros(precc, 4)
        b[3] = (2n+1)/r[end,end] 

        # Solve the system 
        C = M \ b

        # Combine the linearly independent solutions
        # to get the solution vector in each sublayer
        for i in 2:nlayers
            for j in 1:nsublayers-1
                y[:,j,i] = @view(y1_4[:,:,j,i])*C
            end
        end

        return y
    end

    """
        compute_y(r, ρ, g, μ, K; core="liquid")

    Compute the single-phase tidal solution y with the propagator matrix method. 
    
    Returns `y`, an array with size (6, `nr`, `nlay`), where `nr` is the
    number of secondary layers, and `nlay` is the number of primary layers, 
    e.g., core, mantle, crust. The first index corresponds to y1 to y6, where

    y1 = radial displacement \\
    y2 = tangential displacement \\
    y3 = radial stress \\
    y4 = tangential stress \\
    y5 = gravitational potential \\
    y6 = potential stress \\

    # Example
        # Define four layer body
        ρs = [7640, 3300, 3300, 3000.]  # Solid density
        μ =  [60, 60, 60, 60] .* 1e9    # Shear modulus
        κs = [100, 200, 200, 200].*1e9  # Solid bulk modulus
       
        # Discretize primary layers into 100 secondary layers
        rr = expand_layers(r, nr=100)   
        g = get_g(rr, ρs);               # get gravity profile 

        y1 = compute_y(rr, ρ, g, μ, κs);
    """
    function compute_y(r, ρ, g, μ, K; core="liquid")
        r, ρ, g, μ, K = convert_params_to_prec(r, ρ, g, μ, K)

        nlayers = size(r)[2]
        nsublayers = size(r)[1]

        y_start = get_Ic(r[end,1], ρ[1], g[end,1], μ[1], core; M=6, N=3)

        y1_4 = zeros(precc, 6, 3, nsublayers-1, nlayers) # Three linearly independent y solutions
        y = zeros(ComplexF64, 6, nsublayers-1, nlayers)
        
        for i in 2:nlayers
            Bprod = zeros(precc, 6, 6, nsublayers-1)
            @views get_B_product!(Bprod, r[:, i], ρ[i], g[:, i], μ[i], K[i])

            for j in 1:nsublayers-1
                y1_4[:,:,j,i] = @view(Bprod[:,:,j]) * y_start 
            end

            y_start[:,:] .= @view(y1_4[:,:,end,i])   # Set starting vector for next layer
        end

        M = zeros(precc, 3,3)

        M[1, :] .= y1_4[3,:,end,end]  # Row 1 - Radial Stress 
        M[2, :] .= y1_4[4,:,end,end]  # Row 2 - Tangential Stress
        M[3, :] .= y1_4[6,:,end,end]  # Row 3 - Potential Stress
         
        b = zeros(precc, 3)
        b[3] = (2n+1)/r[end,end] 
        C = M \ b

        for i in 2:nlayers
            for j in 1:nsublayers-1
                y[:,j,i] = @view(y1_4[:,:,j,i])*C
            end
        end

        return y
    end

    "Get the core solution vector"
    function get_Ic(r, ρ, g, μ, type; M=6, N=3)
        Ic = zeros(precc, M, N)

        if type=="liquid"
            Ic[1,1] = -r^n / g
            Ic[1,3] = 1.0
            Ic[2,2] = 1.0
            Ic[3,3] = g*ρ
            Ic[5,1] = r^n
            Ic[6,1] = 2(n-1)*r^(n-1)
            Ic[6,3] = 4π * G * ρ 
        else # incompressible solid core
            # First column
            Ic[1, 1] = n*r^( n+1 ) / ( 2*( 2n + 3) )
            Ic[2, 1] = ( n+3 )*r^( n+1 ) / ( 2*( 2n+3 ) * ( n+1 ) )
            Ic[3, 1] = ( n*ρ*g*r + 2*( n^2 - n - 3)*μ ) * r^n / ( 2*( 2n + 3) )
            Ic[4, 1] = n *( n+2 ) * μ * r^n / ( ( 2n + 3 )*( n+1 ) )
            Ic[6, 1] = 2π*G*ρ*n*r^( n+1 ) / ( 2n + 3 )

            # Second column
            Ic[1, 2] = r^( n-1 )
            Ic[2, 2] = r^( n-1 ) / n
            Ic[3, 2] = ( ρ*g*r + 2*( n-1 )*μ ) * r^( n-2 )
            Ic[4, 2] = 2*( n-1 ) * μ * r^( n-2 ) / n
            Ic[6, 2] = 4π*G*ρ*r^( n-1 )

            # Third column
            Ic[3, 3] = -ρ * r^n
            Ic[5, 3] = -r^n
            Ic[6, 3] = -( 2n + 1) * r^( n-1 )

        end

        return Ic
    end

    """
        expand_layers(r; nr::Int=80)

    Discretize the primary layers given by `r` into `nr` discrete secondary layers.

    # Example 
    ```jldoctest
    julia> r = [0, 100., 200.] # two layer body, 100km core and 100km mantle;
    julia> expand_layers(r; nr=5)
    6×2 Matrix{Float64}:
       0.0  100.0
      20.0  120.0
      40.0  140.0
      60.0  160.0
      80.0  180.0
     100.0  200.0
    ```
    """
    function expand_layers(r; nr::Int=80)
        set_nr(nr) # Update nr globally 

        rs = zeros(Float64, (nr+1, length(r)-1))
        
        for i in 1:length(r)-1
            rfine = LinRange(r[i], r[i+1], nr+1)
            rs[:, i] .= rfine[1:end] 
        end
    
        return rs
    end

    """    
        get_total_heating(y, ω, R, ecc)

    Get the total heating rate for solution `y` due to orbital eccentricity `e` 
    for a body with synchronous rotation rate `ω` and radius `R`.
    """
    function get_total_heating(y, ω, R, ecc)
        k2 = y[5, end,end] - 1.0    # Get k2 Love number at surface
        total_power = -21/2 * imag(k2) * (ω*R)^5/G * ecc^2

        return total_power
    end
    
    # 
    """
        get_heating_map(y, r, ρ, g, μ, κ, ω, ecc; vol=false)

    Get a surface heating map for solid-body tides and eccentricity forcing,
    assuming synchronous rotation.
    Heating rate is computed  with numerical integration using the solution `y` returned by [`compute_y`](@ref), 
    using Eq. 2.39a/b integrated in radius. 
    The forcing magnitude is determined by orbital frequency `ω` and eccentricity `ecc`. 
    
    Returns the radially integrated heating maps in W/m^2 for dissipation due to shear 
    and compaction. If `vol=true`, then additional volumentric heating arrays 
    will also be returned in W/m^3.
    """
    function get_heating_map(y, r, ρ, g, μ, κ, ω, ecc; vol=false)
        dres = deg2rad(res)
        λ = κ .- 2μ/3
        R = r[end,end]

        @views clats = Love.clats[:]
        @views lons = Love.lons[:]
        cosTheta = cos.(clats)

        nlats = length(clats)
        nlons = length(lons)
        nsublay = size(r)[1]
        nlay = size(r)[2]

        ϵ = zeros(ComplexF64, nlats, nlons, 6, nsublay-1, nlay)
        ϵs = zero(ϵ)

        n = 2
        ms = [-2, 0, 2]
        forcing = [-1/8, -3/2, 7/8] * ω^2*R^2*ecc 
        for x in eachindex(ms)
            m = ms[x]

            for i in 2:nlay # Loop over primary layers
                ρr = ρ[i]
                κr = κ[i]
                μr = μ[i]
                λr = λ[i]

                for j in 1:nsublay-1 # Loop over sublayers 
                    @views yrr = y[1:6,j,i]
                    
                    rr = r[j,i]
                    gr = g[j,i]

                    compute_strain_ten!(@view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, κr)
                end
            end

            ϵs .+= forcing[x]*ϵ
        end

        Eμ_map = zeros(  (size(ϵ)[1], size(ϵ)[2] ) )
        Eμ = zero(Eμ_map)
        Eμ_vol = zeros(  (size(ϵ)[1], size(ϵ)[2], size(ϵ)[4], size(ϵ)[5] ) )

        Eκ_map = zero(Eμ_map)
        Eκ = zero(Eμ)
        Eκ_vol = zero(Eμ_vol)

        for j in 2:nlay   # loop from CMB to surface
            for i in 1:nsublay-1
                dr = (r[i+1, j] - r[i, j])

                @views Eμ[:,:] .= ω * imag(μ[j]) * (sum(abs.( ϵs[:,:,1:3,i,j] ).^2, dims=3) .+ 2sum(abs.(ϵs[:,:,4:6,i,j]).^2, dims=3) .- 1/3 .* abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2)
                
                @views Eκ[:,:] .= ω/2 *imag(κ[j]) * abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2

                if vol
                    Eμ_vol[:,:,i,j] .= Eμ[:,:]
                    Eκ_vol[:,:,i,j] .= Eκ[:,:]
                end

                Eμ_map[:,:] .+= @view(Eμ[:,:])*dr # Integrate over radius
                Eκ_map[:,:] .+= @view(Eκ[:,:])*dr # Integrate over radius         
            end
        end

        if vol
            return Eμ_map, Eμ_vol, Eκ_map, Eκ_vol
        end 
        
        return Eμ_map, Eκ_map
    end

    """
        function get_heating_map(y, r, ρ, g, μ, Ks, ω, ρl, Kl, Kd, α, ηl, ϕ, k, ecc; vol=false)

    Get a surface heating map for two-phase tides and eccentricity forcing,
    assuming synchronous rotation.
    Heating rate is computed with numerical integration using the solution 
    `y` returned by [`compute_y`](@ref), 
    using Eq. 2.39a/b/c integrated in radius. 
    The forcing magnitude is determined by orbital frequency `ω` and eccentricity `ecc`. 
    
    Returns the radially integrated heating maps in W/m^2 for dissipation due to shear, 
    compaction, and Darcy flow. If `vol=true`, then additional volumentric heating arrays 
    will also be returned in W/m^3.
    """
    function get_heating_map(y, r, ρ, g, μ, Ks, ω, ρl, Kl, Kd, α, ηl, ϕ, k, ecc; vol=false)
        dres = deg2rad(res)
        λ = Kd .- 2μ/3
        R = r[end,end]

        @views clats = Love.clats[:]
        @views lons = Love.lons[:]

        nlats = length(clats)
        nlons = length(lons)
        nsublay = size(r)[1]
        nlay = size(r)[2]

        cosTheta = cos.(clats)

        ϵ       = zeros(ComplexF64, nlats, nlons, 6, nsublay-1, nlay)
        d_disp  = zeros(ComplexF64, nlats, nlons, 3, nsublay-1, nlay)
        p       = zeros(ComplexF64, nlats, nlons, nsublay-1, nlay)

        ϵs = zero(ϵ)
        d_disps = zero(d_disp)
        ps = zero(p)

        n = 2
        ms = [-2, 0, 2]
        forcing = [-1/8, -3/2, 7/8] * ω^2*R^2*ecc 
        for x in eachindex(ms)
            m = ms[x]

            @views Y    = Love.Y[x,:,:]

            for i in 2:nlay # Loop over primary layers
                ρr = ρ[i]
                Ksr = Ks[i]
                μr = μ[i]
                λr = λ[i]
                ρlr = ρl[i]
                Klr = Kl[i]
                Kdr = Kd[i]
                αr = α[i]
                ηlr = ηl[i]
                ϕr = ϕ[i]
                kr = k[i]

                for j in 1:nsublay-1 # Loop over sublayers 
                    @views yrr = y[:,j,i]
                    (y1, y2, y3, y4, y5, y6, y7, y8) = yrr
                    
                    rr = r[j,i]
                    gr = g[j,i]

                    compute_strain_ten!(@view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
                    
                    if ϕ[i] > 0 
                        compute_darcy_displacement!(@view(d_disp[:,:,:,j,i]), yrr, m, rr, ω, ϕr, ηlr, kr, gr, ρlr)
                        compute_pore_pressure!(@view(p[:,:,j,i]), yrr, m)
                    end

                    p[:,:,j,i] .= y7 * Y    # pore pressure
                end
            end

            ϵs      .+= forcing[x]*ϵ
            d_disps .+= forcing[x]*d_disp
            ps      .+= forcing[x]*p
        end

        # Shear heating in the solid
        Eμ_map = zeros(  (nlats, nlons ) )
        Eμ = zero(Eμ_map)
        Eμ_vol = zeros(  (nlats, nlons, nsublay-1, nlay ) )

        Eκ_map = zero(Eμ_map)
        Eκ = zero(Eμ)
        Eκ_vol = zero(Eμ_vol)

        El_map = zero(Eμ_map)
        El = zero(Eμ)
        El_vol = zero(Eμ_vol)

        for j in 2:nlay   # loop from CMB to surface
            for i in 1:nsublay-1
                dr = r[i+1, j] - r[i, j]

                @views Eμ[:,:] .= ω * imag(μ[j]) * (sum(abs.(ϵs[:,:,1:3,i,j]).^2, dims=3) .+ 2sum(abs.(ϵs[:,:,4:6,i,j]).^2, dims=3) .- 1/3 .* abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2)
                @views Eκ[:,:] .= ω/2 *imag(Kd[j]) * abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
    
                if ϕ[j] > 0
                    @views  Eκ[:,:] .+= ω/2 *imag(Kd[j]) * (abs.(ps[:,:,i,j]) ./ Ks[j]).^2
                    @views  El[:,:] .= 0.5 *  ϕ[j]^2 * ω^2 * ηl[j]/k[j] * (abs.(d_disps[:,:,1,i,j]).^2 + abs.(d_disps[:,:,2,i,j]).^2 + abs.(d_disps[:,:,3,i,j]).^2)
                    @views  El_map[:,:] .+= El[:,:]*dr
                end

                if vol
                    @views Eμ_vol[:,:,i,j] .= Eμ[:,:]
                    @views Eκ_vol[:,:,i,j] .= Eκ[:,:]
                    @views El_vol[:,:,i,j] .= El[:,:]
                end

                @views Eμ_map[:,:] .+= Eμ[:,:]*dr # Integrate over radius
                @views Eκ_map[:,:] .+= Eκ[:,:]*dr # Integrate over radius         
            end
        end

        if vol
            return Eμ_map, Eμ_vol, Eκ_map, Eκ_vol, El_map, El_vol
        end 

        return Eμ_map, Eκ_map, El_map
    end


    """
        function get_heating_profile(y, r, ρ, g, μ, κ, ω, ecc; lay=nothing)

    Get the radial volumetric heating for solid-body tides and eccentricity forcing,
    assuming synchronous rotation.
    Heating rate is computed with numerical integration using the solution 
    `y` returned by [`compute_y`](@ref), 
    using Eq. 2.39a/b integrated over solid angle. 
    The forcing magnitude is determined by orbital frequency `ω` and eccentricity `ecc`. 
    The heating profile for a specific layer is specified with `lay`, otherwise all 
    layers will be caclulated.

    Returns the angular averaged volumetric heating profiles in W/m^3 for dissipation due to shear 
    and compaction, as well as the total power dissipated in each primary layer.
    """
    function get_heating_profile(y, r, ρ, g, μ, κ, ω, ecc; lay=nothing)
        dres = deg2rad(res)
        R = r[end,end]

        @views clats = Love.clats[:]
        @views lons = Love.lons[:]

        nsublay = size(r)[1]
        nlay = size(r)[2]
        nlats = length(clats)
        nlons = length(lons)

        cosTheta = cos.(clats)

        ϵ = zeros(ComplexF64, nlats, nlons, 6, nsublay, nlay)
        ϵs = zero(ϵ)

        n = 2
        ms = [-2, 0, 2]
        forcing = [-1/8, -3/2, 7/8] * ω^2*R^2*ecc 

        # Retrieve stain tensor 
        for x in eachindex(ms)
            m = ms[x]

            for i in 2:nlay # Loop of layers
                ρr = ρ[i]
                κr = κ[i]
                μr = μ[i]

                for j in 1:nsublay-1 # Loop over sublayers 
                    @views yrr = y[1:6,j,i]
                    
                    rr = r[j,i]
                    gr = g[j,i]

                    compute_strain_ten!(@view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, κr)
                end
            end

            ϵs .+= forcing[x]*ϵ
        end

        Eμ     = zeros(  (nlats, nlons, nsublay-1, nlay) )
        Eμ_tot = zeros(  (nlay) )     # Total heating rate (W)
        Eμ_vol = zeros(  size(r) )    # Spherical avg. volumetric heating rate in each layer (W/m^3)

        Eκ = zero(Eμ)
        Eκ_tot = zero( Eμ_tot )
        Eκ_vol = zero( Eμ_vol )

        if isnothing(lay)
            rstart = 2
            rend = nlay
        else
            rstart = lay
            rend = lay
        end

        # Compute dissipation
        for j in rstart:rend    
            for i in 1:nsublay-1
                dr = (r[i+1, j] - r[i, j])
                dvol = 4π/3 * (r[i+1, j]^3 - r[i, j]^3)

                @views Eμ[:,:,i, j] = sum(abs.(ϵs[:,:,1:3,i,j]).^2, dims=3) .+ 2sum(abs.(ϵs[:,:,4:6,i,j]).^2, dims=3) .- 1/3 .* abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
                Eμ[:,:,i, j] .*= ω * imag(μ[j])
    
                @views Eκ[:,:,i, j] = ω/2 *imag(κ[j]) * abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2

                # Integrate across r to find dissipated energy per unit area
                Eμ_vol[i,j] = sum(sin.(clats) .* (Eμ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
                Eμ_tot[j] += Eμ_vol[i,j]*dvol
    
                Eκ_vol[i,j] = sum(sin.(clats) .* (Eκ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
                Eκ_tot[j] += Eκ_vol[i,j]*dvol
            end
        end

        return (Eμ_tot, Eμ_vol), (Eκ_tot, Eκ_vol) 
    end

    """
        get_heating_profile(y, r, ρ, g, μ, Ks, ω, ρl, Kl, Kd, α, ηl, ϕ, k, ecc; lay=nothing)

    Get the radial volumetric heating for two-phase tides and eccentricity forcing,
    assuming synchronous rotation.
    Heating rate is computed with numerical integration using the solution 
    `y` returned by [`compute_y`](@ref), 
    using Eq. 2.39a/b/c integrated over solid angle. 
    The forcing magnitude is determined by orbital frequency `ω` and eccentricity `ecc`. 
    The heating profile for a specific layer is specified with `lay`, otherwise all 
    layers will be caclulated.

    Returns the angular averaged volumetric heating profiles in W/m^3 for dissipation due to shear, 
    compaction and Darcy flow, as well as the total power dissipated in each primary layer.
    """
    function get_heating_profile(y, r, ρ, g, μ, Ks, ω, ρl, Kl, Kd, α, ηl, ϕ, k, ecc; lay=nothing)
        dres = deg2rad(res)
        R = r[end,end]

        @views clats = Love.clats[:]
        @views lons = Love.lons[:]

        nsublay = size(r)[1]
        nlay = size(r)[2]
        nlats = length(clats)
        nlons = length(lons)

        cosTheta = cos.(clats)

        ϵ = zeros(ComplexF64, nlats, nlons, 6, nsublay-1, nlay)
        d_disp = zeros(ComplexF64, nlats, nlons, 3, nsublay-1, nlay)
        p = zeros(ComplexF64, nlats, nlons, nsublay-1, nlay)

        ϵs = zero(ϵ)
        d_disps = zero(d_disp)
        ps = zero(p)

        n = 2
        ms = [-2, 0, 2]
        forcing = [-1/8, -3/2, 7/8] * ω^2*R^2*ecc 
        for x in eachindex(ms)
            m = ms[x]

            @views Y    = Love.Y[x,:,:]

            for i in 2:nlay # Loop of layers
                ρr = ρ[i]
                Ksr = Ks[i]
                μr = μ[i]
                ρlr = ρl[i]
                Klr = Kl[i]
                Kdr = Kd[i]
                αr = α[i]
                ηlr = ηl[i]
                ϕr = ϕ[i]
                kr = k[i]

                for j in 1:nsublay-1 # Loop over sublayers 
                    @views yrr = y[:,j,i]
                    (y1, y2, y3, y4, y5, y6, y7, y8) = yrr
                    
                    rr = r[j,i]
                    gr = g[j,i]

                    compute_strain_ten!(@view(ϵ[:,:,:,j,i]), yrr, m, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
                    
                    if ϕ[i] > 0 
                        compute_darcy_displacement!(@view(d_disp[:,:,:,j,i]), yrr, m, rr, ω, ϕr, ηlr, kr, gr, ρlr)
                        compute_pore_pressure!(@view(p[:,:,j,i]), yrr, m)
                    end

                    p[:,:,j,i] .= y7 * Y    # pore pressure

                end
            end

            ϵs .+= forcing[x]*ϵ
            d_disps .+= forcing[x]*d_disp
            ps .+= forcing[x]*p

        end

        # Shear heating in the solid
        Eμ = zeros(  (size(ϵ)[1], size(ϵ)[2], size(ϵ)[4], size(ϵ)[5]) )
        Eμ_tot = zeros(  (nlay) )
        Eμ_vol = zeros(  size(r) )

        # Darcy dissipation in the liquid
        El = zeros(  (size(ϵ)[1], size(ϵ)[2], size(ϵ)[4], size(ϵ)[5]) )
        El_tot = zeros(  (nlay) )
        El_vol = zeros(  size(r) )

        # Bulk dissipation in the solid
        Eκ = zero(Eμ)
        Eκ_tot = zero( Eμ_tot )
        Eκ_vol = zero( Eμ_vol )


        if isnothing(lay)
            rstart = 2
            rend = 4
        else
            rstart = lay
            rend = lay
        end

        for j in rstart:rend   # loop from CMB to surface          
            for i in 1:nsublay-1
                dr = (r[i+1, j] - r[i, j])
                dvol = 4π/3 * (r[i+1, j]^3 - r[i, j]^3)

                @views Eμ[:,:,i, j] = sum(abs.(ϵs[:,:,1:3,i,j]).^2, dims=3) .+ 2sum(abs.(ϵs[:,:,4:6,i,j]).^2, dims=3) .- 1/3 .* abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
                Eμ[:,:,i, j] .*= ω * imag(μ[j])

                @views Eκ[:,:,i, j] = ω/2 *imag(Kd[j]) * abs.(sum(ϵs[:,:,1:3,i,j], dims=3)).^2
                if ϕ[j] > 0
                    @views Eκ[:,:,i, j] .+= ω/2 *imag(Kd[j]) * (abs.(ps[:,:,i,j]) ./ Ks[j]).^2
                end

                # Integrate across r to find dissipated energy per unit area
                @views Eμ_vol[i,j] = sum(sin.(clats) .* (Eμ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
                Eμ_tot[j] += Eμ_vol[i,j]*dvol

                @views Eκ_vol[i,j] = sum(sin.(clats) .* (Eκ[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
                Eκ_tot[j] += Eκ_vol[i,j]*dvol
        
                if ϕ[j] > 0            
                    @views El[:,:,i, j] = 0.5 *  ϕ[j]^2 * ω^2 * ηl[j]/k[j] * (abs.(d_disps[:,:,1,i,j]).^2 + abs.(d_disps[:,:,2,i,j]).^2 + abs.(d_disps[:,:,3,i,j]).^2)
                    @views El_vol[i,j] = sum(sin.(clats) .* (El[:,:,i,j])  * dres^2) * r[i,j]^2.0 * dr / dvol
                    El_tot[j] += El_vol[i,j]*dvol

                end

            end
        end

        return (Eμ_tot, Eμ_vol), (Eκ_tot, Eκ_vol), (El_tot, El_vol)
    end
#
    """
        define_spherical_grid(res)

    Create the spherical grid of angular resolution `res` in degrees. This is used for 
    numerical integrations over solid angle. A new grid can easily be defined by 
    recalling the function with a new `res`.

    The grid is internal to Love, but can be accessed with 

        Love.clats[:] # colatitude grid
        Love.lons[:]  # longitude grid
    """
    function define_spherical_grid(res; n=2)
        Love.res = res

        lons = deg2rad.(collect(0:res:360-0.001))'
        clats = deg2rad.(collect(0:res:180))
        clats[1] += 1e-6
        clats[end] -= 1e-6
        cosTheta = cos.(clats)

        Love.Y    = zeros(ComplexF64, 3, length(clats), length(lons))
        Love.dYdθ = zero(Love.Y)
        Love.dYdϕ = zero(Love.Y)
        Love.Z    = zero(Love.Y)
        Love.X    = zero(Love.Y)

        ms = [-2, 0, 2]
        for i in eachindex(ms)
            m = ms[i]
            Love.Y[i,:,:] .= m < 0 ? Ynmc(n,abs(m),clats,lons) : Ynm(n,abs(m),clats,lons)
            # Love.Y[i,:,:] .= Ynm(n,abs(m),clats,lons)

            if iszero(abs(m))
                Love.dYdθ[i,:,:] = -1.5sin.(2clats) * exp.(1im * m * lons)
                Love.dYdϕ[i,:,:] = Love.Y[i,:,:] * 1im * m

                Love.Z[i,:,:] = 0.0 * Love.Y[i,:,:]
                Love.X[i,:,:] = -6cos.(2clats)*exp.(1im *m * lons) .+ n*(n+1)*Love.Y[i,:,:]

            elseif  abs(m) == 2
                Love.dYdθ[i,:,:] = 3sin.(2clats) * exp.(1im * m * lons)
                Love.dYdϕ[i,:,:] = Y[i,:,:] * 1im * m
                
                Love.Z[i,:,:] = 6 * 1im * m * cos.(clats) * exp.(1im * m * lons)
                Love.X[i,:,:] = 12cos.(2clats)* exp.(1im * m * lons) .+ n*(n+1)*Y[i,:,:] 
            end

        end

        Love.clats = clats;
        Love.lons = lons;

    end

    """
        compute_strain_ten!(ϵ, y, m, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)

        Calculate the strain tensor ϵ at a particular radial level. 
    """
    function compute_strain_ten!(ϵ, y, m, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
        n = 2.0
        i = get_m_ind(m)

        @views Y    = Love.Y[i,:,:]
        @views dYdθ = Love.dYdθ[i,:,:]
        @views dYdϕ = Love.dYdϕ[i,:,:]
        @views Z    = Love.Z[i,:,:]
        @views X    = Love.X[i,:,:]

        y1 = y[1]
        y2 = y[2]
        y3 = y[3]
        y4 = y[4]
        y7 = y[7]

        λr = Kdr - 2μr/3
        βr = λr + 2μr

        # Compute strain tensor
        ϵ[:,:,1] = (-2λr*y1 + n*(n+1)λr*y2 + rr*y3 + rr*αr*y7)/(βr*rr)  * Y
        ϵ[:,:,2] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y + 0.5y2*X)
        ϵ[:,:,3] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y - 0.5y2*X)
        ϵ[:,:,4] = 0.5/μr * y4 * dYdθ        
        ϵ[:,:,5] = 0.5/μr * y4 * dYdϕ .* 1.0 ./ sin.(clats) 
        ϵ[:,:,6] = 0.5 * y2/rr * Z
    end

    """
        compute_stress_ten!(σ, ϵ, y, m, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)

        Calculate the stress tensor σ at a particular radial level. 
    """
    function compute_stress_ten!(σ, ϵ, y, m, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
        i = get_m_ind(m)

        @views Y    = Love.Y[i,:,:]

        y1 = y[1]
        y2 = y[2]
        y7 = y[7]

        A = get_A(rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)
        dy1dr = dot(A[1,:], y[:])

        λr = Ksr .- 2μr/3

        ϵV = dy1dr .+ 2/rr * y1 .- n*(n+1)/rr * y2
        F = (λr * ϵV .- αr*y7) .* Y
        σ[:,:,1] .= F .+ 2μr*ϵ[:,:,1]  
        σ[:,:,2] .= F .+ 2μr*ϵ[:,:,2] 
        σ[:,:,3] .= F .+ 2μr*ϵ[:,:,3] 
        σ[:,:,4] .= 2μr * ϵ[:,:,4]
        σ[:,:,5] .= 2μr * ϵ[:,:,5]
        σ[:,:,6] .= 2μr * ϵ[:,:,6]
    end

    """
        compute_strain_ten!(ϵ, y, m, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)

        Calculate the strain tensor ϵ at a particular radial level. 
    """
    function compute_strain_ten!(ϵ, y, m, rr, ρr, gr, μr, Ksr)
        n = 2.0
        i = get_m_ind(m)

        @views Y    = Love.Y[i,:,:]
        @views dYdθ = Love.dYdθ[i,:,:]
        @views dYdϕ = Love.dYdϕ[i,:,:]
        @views Z    = Love.Z[i,:,:]
        @views X    = Love.X[i,:,:]

        y1 = y[1]
        y2 = y[2]
        y3 = y[3]
        y4 = y[4]

        λr = Ksr .- 2μr/3
        βr = λr + 2μr

        # Compute strain tensor
        ϵ[:,:,1] = (-2λr*y1 + n*(n+1)λr*y2 + rr*y3)/(βr*rr)  * Y
        ϵ[:,:,2] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y + 0.5y2*X)
        ϵ[:,:,3] = 1/rr * ((y1 - 0.5n*(n+1)y2)Y - 0.5y2*X)
        ϵ[:,:,4] = 0.5/μr * y4 * dYdθ        
        ϵ[:,:,5] = 0.5/μr * y4 * dYdϕ .* 1.0 ./ sin.(clats) 
        ϵ[:,:,6] = 0.5 * y2/rr * Z
    end

    """
        compute_stress_ten!(σ, ϵ, y, m, rr, ρr, gr, μr, Ksr, ω, ρlr, Klr, Kdr, αr, ηlr, ϕr, kr)

        Calculate the stress tensor σ at a particular radial level. 
    """
    function compute_stress_ten!(σ, ϵ, y, m, rr, ρr, gr, μr, κsr)
        n = 2
        i = get_m_ind(m)

        @views Y    = Love.Y[i,:,:]

        y1 = y[1]
        y2 = y[2]
        y3 = y[3]

        λr = κsr .- 2μr/3
        βr = λr + 2μr

        ϵV = (4μr*y1 .+ rr * y3 .- 2n*(n+1)*μr * y2)/(βr*rr)

        F = (λr * ϵV) .* Y
        σ[:,:,1] .= y3 .* Y 
        σ[:,:,2] .= F .+ 2μr*ϵ[:,:,2] 
        σ[:,:,3] .= F .+ 2μr*ϵ[:,:,3] 
        σ[:,:,4] .= 2μr * ϵ[:,:,4]
        σ[:,:,5] .= 2μr * ϵ[:,:,5]
        σ[:,:,6] .= 2μr * ϵ[:,:,6]
    end

    """
        compute_displacement!(dis, y, m)

        Calculate the solid displacement vector at a particular radial level. 
    """
    function compute_displacement!(dis, y, m)
        i = get_m_ind(m)

        @views Y    = Love.Y[i,:,:]
        @views dYdθ = Love.dYdθ[i,:,:]
        @views dYdϕ = Love.dYdϕ[i,:,:]

        y1 = y[1]
        y2 = y[2]

        dis[:,:,1] =  Y * y1
        dis[:,:,2] = dYdθ * y2
        dis[:,:,3] = dYdϕ * y2 ./ sin.(clats)
    end

    """
        compute_darcy_displacement!(dis_rel, y, m, r, ω, ϕ, ηl, k, g, ρl)

        Calculate the Darcy displacement vector at a particular radial level. 
    """
    function compute_darcy_displacement!(dis_rel, y, m, r, ω, ϕ, ηl, k, g, ρl)
        i = get_m_ind(m)

        @views Y    = Love.Y[i,:,:]
        @views dYdθ = Love.dYdθ[i,:,:]
        @views dYdϕ = Love.dYdϕ[i,:,:]
        
        y1 = y[1]
        y5 = y[5]
        y7 = y[7]
        y8 = y[8]
        y9 = 1im * k / (ω*ϕ*ηl*r) * (ρl*g*y1 - ρl * y5 + ρl*g*y8 + y7)
        
        dis_rel[:,:,1] = Y * y8 
        dis_rel[:,:,2] = dYdθ * y9
        dis_rel[:,:,3] = dYdϕ * y9 ./ sin.(clats)
    end

    """
        compute_pore_pressure!(p, y, m)

        Calculate the fluid pore pressur at a particular radial level. 
    """
    function compute_pore_pressure!(p, y, m)
        i = get_m_ind(m)

        @views Y    = Love.Y[i,:,:]
        @views dYdθ = Love.dYdθ[i,:,:]
        @views dYdϕ = Love.dYdϕ[i,:,:]
        
        y7 = y[7]
        
        p[:,:] = Y * y7 
    end

    # Compute the spherical harmonic coeffs for isotropic components vs radius
    function get_radial_isotropic_coeffs(y, r, ρ, g, μ, Ks, ω, ρl, Kl, Kd, α, ηl, ϕ, k)
        ϵsV = zeros(ComplexF64, size(r)[1]-1, size(r)[2])
        ϵrelV = zero(ϵsV) 
        pl = zero(ϵsV)

        y1 = @view(y[1,:,:])
        y2 = @view(y[2,:,:])
        y3 = @view(y[3,:,:])
        y4 = @view(y[4,:,:])
        y7 = @view(y[7,:,:])

        for i in 2:size(r)[2] # Loop over layers
            ρr = ρ[i]
            Kdr = ϕ[i] > 0 ? Kd[i] : Ks[i]
            μr = μ[i]
            αr = α[i]
            ϕr = ϕ[i]
            λr = Kdr .- 2μr/3
            βr = λr + 2μr
            S = ϕr / Kl[i] + (αr - ϕr) / Ks[i]      # Storativity 

            for j in 1:size(r)[1]-1 # Loop over sublayers 
                rr = r[j,i]

                # Compute strain tensor
                ϵsV[j,i] = (4μr*y1[j,i] + rr*y3[j,i] - 2n*(n+1)*μr*y2[j,i] + αr*rr*y7[j,i])/(βr*rr) 
                ϵrelV[j,i] = - αr/ϕr * (ϵsV[j,i] + y7[j,i]/αr * S )
                pl[j,i]   = y7[j,i]
            end
        end

        return ϵsV, ϵrelV, pl

    end

        # Compute the spherical harmonic coeffs for isotropic components vs radius
    function get_radial_isotropic_coeffs(y, r, ρ, g, μ, Ks)
        ϵsV = zeros(ComplexF64, size(r)[1]-1, size(r)[2])
        # ϵrelV = zero(ϵsV) 
        # pl = zero(ϵsV)

        y1 = y[1,:,:]
        y2 = y[2,:,:]
        y3 = y[3,:,:]   
        y4 = y[4,:,:]

        for i in 2:size(r)[2] # Loop over layers
            ρr = ρ[i]
            Ksr = Ks[i]
            μr = μ[i]
            λr = Ksr .- 2μr/3
            βr = λr + 2μr

            for j in 1:size(r)[1]-1 # Loop over sublayers 
                rr = r[j,i]

                # Compute strain tensor
                ϵsV[j,i] = (4μr*y1[j,i] + rr*y3[j,i] - 2n*(n+1)*μr*y2[j,i])/(βr*rr) 
                # ϵrelV[j,i] = - αr/ϕr * (ϵsV[j,i] + y7[j,i]/αr * S )
                # pl[j,i]   = y7[j,i]
            end
        end

        return ϵsV#, ϵrelV, pl

    end


    function get_ke_power(y, r, ρ, g, μ, Ks, ω, ρl, Kl, Kd, α, ηl, ϕ, k)
        n = 2
        y1 = y[1]
        y5 = y[5]   
        y7 = y[7]
        y8 = y[8]

        # println(y1, " ", y5, " ", y7, " ", y8)

        y9 = 1im * k ./ (ω*ϕ*ηl*r) .* (ρl*g*y1 .- ρl * y5 .+ ρl*g .* y8 .+ y7)

        return y8, y9, abs.(y8).^2 .+ n*(n+1)*abs.(y9).^2  
    end

    function get_m_ind(m)
        if m == -2
            i=1
        elseif m == 0
            i=2
        elseif m == 2
            i=3
        else
            error("m must be -2, 0, or 2")
        end

        return i
    end
end