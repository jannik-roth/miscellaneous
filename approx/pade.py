import sympy as smp

# Generates Pade approximatin using sympy
# taken from: https://rwdb.xyz/pad%C3%A9-approximations/

def pade(f, x, n, m, leading_denumer=1):
    a = smp.symbols("a:{}".format(n + 1))
    b = smp.symbols("b:{}".format(m + 1))
    P = sum(a[k] * x**k for k in range(n + 1))
    Q = sum(b[k] * x**k for k in range(m + 1)).subs(b[0], leading_denumer)

    expansion = smp.series(f * Q - P, x, n=n + m + 1)
    eqns = [expansion.coeff(x, k) for k in range(n + m + 1)]
    sols = smp.solve(eqns)

    return (P / Q).subs(sols)

def diagonal_pade(f, x, n, *args, **kwargs):
    return pade(f, x, n, n, *args, **kwargs)
