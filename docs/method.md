# Method

Implementation conventions for `src/modpinn`; see the
[article](https://doi.org/10.1088/2632-2153/ae5459) for the broader study.

## Fields

```math
 ds^2=-\alpha(t,r)^2\,dt^2+a(t,r)^2\,dr^2+r^2d\Omega^2,
 \qquad G=c=1.
```

Inputs: $`(t,r)`$. Outputs: scalar field $`\phi`$, lapse $`\alpha`$,
and compactness $`C=2m/r=1-a^{-2}`$ (`Q` in code). The scalar is unrestricted;
trainable sigmoids bound lapse and compactness to the unit interval.
Recover $`a=(1-C)^{-1/2}`$ where $`C\lt 1`$.

```math
 \Phi=\partial_r\phi,\qquad
 \Pi=\frac{\partial_t\phi}{\alpha\sqrt{1-C}},\qquad
 m=\frac{rC}{2}.
```

Automatic differentiation supplies derivatives, including second scalar
derivatives. Each model fits one initial amplitude, which is not a network input.

## Residuals and boundaries

```math
\begin{aligned}
R_{1a}&=\partial_t\Phi-\partial_r(\alpha\sqrt{1-C}\,\Pi),\\[2pt]
R_{1b}&=\partial_t\Pi-\frac{1}{r^2}\partial_r(r^2\alpha\sqrt{1-C}\,\Phi),\\[2pt]
R_2&=r(1-C)\partial_r\ln\alpha-\frac r2\partial_r C-C,\\[2pt]
R_3&=\partial_r m-2\pi r^2(1-C)(\Phi^2+\Pi^2).
\end{aligned}
```

Interior density: $`E=R_{1a}^2+R_{1b}^2+R_2^2+R_3^2`$.
$`R_{1a}`$ is a mixed-derivative identity retained for numerical compatibility.
Residual compactness is clamped to $`[0,0.999999]`$ before roots and derivatives;
[evaluation](results.md) uses a different clamp.

Initial loss uses reference scalar/metric profiles and zero momentum, with
squared errors in all four fields plus a scalar-maximum penalty. A missing final
initial-array value takes its preceding value; interior gaps remain unfilled.
Left-boundary fields match reference arrays. Outer residuals:

```math
 r\partial_r\phi+\phi,\qquad
 r\partial_r\alpha+\alpha-1,\qquad
 r\partial_r C+C.
```

Weighted boundary losses replace the first/last radial residual columns.
Interior reference values supply diagnostics only.

## Causal weights

After boundary substitution, index $`E_{ij}`$ by time $`i`$ and radius $`j`$:

```math
 w^r_{ij}=\exp\!\left(-\epsilon_r\sum_{k\lt j}E_{ik}\right).
```

For later slices, $`L_i=N_r^{-1}\sum_j w^r_{ij}E_{ij}`$. At the initial
slice, replace weighted residual entries with initial-condition loss before averaging.

```math
 w^t_i=\exp\!\left(-\epsilon_t\sum_{k\lt i}L_k\right),\qquad
 \mathcal L=\frac1{N_t}\sum_i w^t_iL_i.
```

Weights remain differentiable; sums have no grid-spacing factors. Legacy
per-equation weights are inactive. Boundary/initial weights are set in [training](training.md).

## Network

Five normalized features (time, radius, their squares and product), two learned warps, and 32
physical-coordinate Gaussian RBFs form 39 features, projected to width 64.
Each warp combines a linear term with 32 trainable tanh terms. RBF centers and
anisotropic scales are learned; softplus keeps scales positive.
Quadratic layers compute:

```math
 h(x)=(W_1x+b)\odot(1+W_2x).
```

Hidden outputs use trainable tanh; final outputs use the heads above.
`number_hidden` counts additional layers after the first embedding-to-hidden layer.

## Sampling and optimization

The initial tensor grid is time-major. Radial points shift by
$`(r_{\max}-r_{\min})/N_r`$, avoiding zero but exceeding the nominal upper
bound at the last point. Remeshing clamps to reference bounds, using:

```math
 M(r)=1+\Lambda\frac{E(r)-E_{\min}}{\max(E_{\max}-E_{\min},10^{-8})}.
```

Per time slice, trapezoidal masses define a cumulative distribution;
piecewise-linear inversion places equally spaced mass targets. Point count,
time coordinates, and the initial slice stay fixed. Configuration sets the
start and interval. Remeshing precedes SOAP except in the ablation case, where
it follows the update using that step's previously computed residuals.

Training uses SOAP and clips the gradient norm to one; parameter registration order is retained.
CPU fixtures check derivatives, residuals, remeshing, and three updates.
Training saves the latest state; bundled [results](results.md) use selected states.
