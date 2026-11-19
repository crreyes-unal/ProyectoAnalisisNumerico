### Red neuronal y entrenamiento en Julia

Se muestra la implementación y el entrenamiento de un modelo de difusión basado en una arquitectura WaveNet simplificada utilizando Flux.jl.

### 0. Dependencias y configuración inicial

Importación de librerías necesarias.

```julia
using Flux
using CUDA
using Statistics
using Printf
using Optimisers

device = CUDA.functional() ? gpu : cpu
println("Dispositivo: ", CUDA.functional() ? "CUDA (GPU)" : "CPU")
```
La última parte del código sirve para verificar si hay soporte para GPU mediante el paquete CUDA.

### 1. Datos y esquema de ruido

Se definen los parámetros de la señal. Además corresponde al proceso directo.
440 Hz correspondientes a la nota La.

```julia
const FS = 16000
const F0 = 440f0
const LARGO = 2048

function lote_pitidos(B, largo=LARGO)
    n = Float32.(0:largo-1)
    # fase: (1, 1, B) para hacer broadcast con la longitud
    fase = 2f0 * π * rand(Float32, 1, 1, B) 
    # reshape crea (LARGO, 1, 1), generando un tensor (LARGO, 1, B)
    x = sin.(2f0 * π * (F0 / FS) .* reshape(n, :, 1, 1) .+ fase)
    return x |> device
end

const T_pasos = 200
# Equivalente a torch.linspace
const beta = collect(range(1f-4, 5f-2, length=T_pasos)) |> device
const alfa = 1f0 .- beta
const abar = cumprod(alfa, dims=1)
@printf("abar_T = %.4f   (con beta_fin=0.05)\n", abar[end])
```

### 2. Arquitectura de la red (`WaveNetMini`)

Red reuronal. Arquitectura de WaveNet simplificada.

```julia
function emb_tiempo(t, dim=64)
    mitad = dim ÷ 2
    frec = exp.(-log(10000f0) .* Float32.(0:mitad-1) ./ (mitad - 1)) |> device
    # t es un vector columna de (B,). Transponemos para multiplicar: (mitad,) * (1, B)
    arg = frec * Float32.(t)' 
    return vcat(sin.(arg), cos.(arg)) # Retorna (dim, B)
end

struct Bloque
    lin_t
    conv
    proy
end
Flux.@functor Bloque

function Bloque(C::Int, dil::Int, dim_t::Int)
    lin_t = Dense(dim_t => C)
    # En Flux, para convoluciones 1D con kernel 3, padding=dil mantiene la longitud original
    conv = Conv((3,), C => 2C, pad=dil, dilation=dil)
    proy = Conv((1,), C => C)
    return Bloque(lin_t, conv, proy)
end

function (m::Bloque)(x, te)
    # x: (L, C, B) | te: (dim_t, B). Reformateamos el tiempo a (1, C, B)
    te_lin = reshape(m.lin_t(te), 1, size(x, 2), size(x, 3))
    
    h = m.conv(x .+ te_lin)
    
    # Dividimos por la dimensión de canales (dim=2)
    a, b = Flux.chunk(h, 2, dims=2)
    h_gated = tanh.(a) .* σ.(b) # Activación Gated
    
    return (x .+ m.proy(h_gated)) .* Float32(sqrt(0.5))
end

struct WaveNetMini
    dim_t::Int
    entrada
    mlp_t
    bloques
    salida
    campo_receptivo::Int
end
Flux.@functor WaveNetMini (entrada, mlp_t, bloques, salida)

function WaveNetMini(; C=32, capas=8, dim_t=64)
    entrada = Conv((1,), 1 => C)
    # x -> x .* σ.(x) es el equivalente exacto a SiLU
    mlp_t = Chain(Dense(dim_t => dim_t), x -> x .* σ.(x)) 
    bloques = [Bloque(C, 2^k, dim_t) for k in 0:capas-1]
    salida = Chain(Conv((1,), C => C, x -> x .* σ.(x)), Conv((1,), C => 1))
    campo_receptivo = 1 + 2 * (2^capas - 1)
    
    return WaveNetMini(dim_t, entrada, mlp_t, bloques, salida, campo_receptivo)
end

function (m::WaveNetMini)(x, t)
    te = m.mlp_t(emb_tiempo(t, m.dim_t))
    h = m.entrada(x)
    for bloque in m.bloques
        h = bloque(h, te)
    end
    return m.salida(h)
end
```

#### Verificación del campo receptivo

Comprobación del número de muestras que abarca el campo receptivo del modelo según las capas de dilatación.

```julia
println("Periodo del pitido: ", round(FS / F0, digits=1), " muestras")
for capas in (3, 8)
    modelo_tmp = WaveNetMini(capas=capas)
    println("capas=capas: campo receptivo = (modelo_tmp.campo_receptivo) muestras")
end
```

### 3. Entrenamiento

```julia
function entrenar(; capas=8, C=32, pasos=8000, B=32, lr_inicial=5f-4)
    modelo = WaveNetMini(C=C, capas=capas) |> device
    
    # Optimizador encadenado: recorte normativo de gradientes seguido de Adam
    reglas_opt = OptimiserChain(ClipNorm(1.0), Adam(lr_inicial))
    opt_state = Flux.setup(reglas_opt, modelo)
    
    hist = Float32[]
    println("Entrenando: capas=\$capas, C=C, campo receptivo=(modelo.campo_receptivo), " *
            "parámetros=\$(sum(length, Flux.params(modelo)))")
    
    for paso in 1:pasos
        x0 = lote_pitidos(B)
        t = rand(1:T_pasos, B) |> device # 1-indexed en Julia
        eps = randn(Float32, size(x0)...) |> device
        
        # Reformateamos abar a (1, 1, B) para broadcastear sobre la secuencia de longitud 2048
        a = reshape(abar[t], 1, 1, B)
        xt = sqrt.(a) .* x0 .+ sqrt.(1f0 .- a) .* eps
        
        loss, grads = Flux.withgradient(modelo) do m
            pred = m(xt, t)
            Flux.mse(pred, eps)
        end
        
        Flux.update!(opt_state, modelo, grads[1])
        
        # Implementación manual de CosineAnnealingLR (T_max=pasos, eta_min=1e-5)
        lr_actual = 1f-5 + 0.5f0 * (lr_inicial - 1f-5) * (1f0 + cos(Float32(π) * paso / pasos))
        Optimisers.adjust!(opt_state, lr_actual) # Actualiza la tasa de forma dinámica
        
        push!(hist, loss)
        if paso % 500 == 0
            media_loss = mean(hist[end-499:end])
            @printf("paso %d/%d | pérdida (media de los últimos 500): %.4f\n", paso, pasos, media_loss)
        end
    end
    return modelo, hist
end

modelo8, hist8 = entrenar(capas=8)
```

### 4. Evaluación de la pérdida por paso temporal $t$

Comparación del error cuadrático medio (MSE) de la red entrenada frente a una línea base trivial en distintos instantes de ruido.

```julia
function perdida_por_t(modelo; ts=(5, 25, 50, 100, 150, 200), B=64)
    modelo_eval = Flux.testmode!(modelo) # Fija parámetros como equivalente a modelo.eval()
    println("  t | pérdida de la red | predicción trivial")
    for tv in ts
        x0 = lote_pitidos(B)
        t = fill(tv, B) |> device
        eps = randn(Float32, size(x0)...) |> device
        
        a = abar[tv]
        xt = sqrt(a) .* x0 .+ sqrt(1f0 .- a) .* eps
        
        pred = modelo_eval(xt, t)
        mse = Flux.mse(pred, eps)
        
        triv = 0.5f0 * a / (1f0 - a)
        @printf("%3d | %17.4f | %.4f\n", tv, mse, triv)
    end
end

perdida_por_t(modelo8)
```

