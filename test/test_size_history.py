
@pytest.fixture
def eta():
    return SizeHistory(t=np.array([np.inf]), Ne=np.array([1.]))

@pytest.fixture
def rnd_eta():
    T = 10
    t = np.r_[0., np.cumsum(np.random.rand(T - 1)), np.inf]
    Ne = np.random.rand(T)
    return SizeHistory(t=t, Ne=Ne)

def test_sim1():
    n = 10
    data = msp.simulate(
        sample_size=n, length=1_000_000, recombination_rate=1e-9, mutation_rate=1.4e-8, Ne=1e4
    )
    xs = XSMC(data, 0, [1], rho_over_theta=1e-9 / 1.4e-8)
    map_path = xs.viterbi()

def test_sim2(eta):
    n = 5
    L = int(1e5)
    sim = msp.simulate(
        sample_size=n, length=L, recombination_rate=.1, mutation_rate=1, Ne=1e-2
    )
    cps = viterbi_path(sim, eta, 1, [0, 2, 3], .1, .1)
    print(cps)

def test_gknns(eta):
    n = 10
    L = int(1e7)
    sim = msp.simulate(
        sample_size=n, length=L, recombination_rate=1e-8, mutation_rate=1e-8, Ne=1e4
    )
    viterbi_path(sim, eta, 0, list(range(1, 10)), 1., 1., gknns=3)

def test_size_history_call(rnd_eta):
    p = rnd_eta
    q = PPoly(x=p.t, c=[1. / p.Ne])
    for t in np.random.rand(100) * len(p.t):
        assert abs(p(t) - q(t)) < 1e-4

def test_size_history_R(rnd_eta):
    p = rnd_eta
    q = PPoly(x=p.t, c=[1. / p.Ne]).antiderivative()
    for t in np.random.rand(100) * len(p.t):
        assert abs(p.R(0., t) - q(t)) < 1e-4
    for t1, t2 in np.random.rand(100, 2) * len(p.t):
        assert abs(p.R(t1, t1 + t2) - (q(t1 + t2) - q(t1))) < 1e-4

def test_size_history_Rn0(rnd_eta):
    p = rnd_eta
    q0 = PPoly(x=p.t, c=[1. / p.Ne])
    q0R = q0.antiderivative()
    def q(t, n0):
        def f(s):
            return n0 * q0(s) / (n0 + (1 - n0) * np.exp(-q0R(s)))
        return quad(f, 0., t, points=p.t[:-1])[0]
    for n0 in range(1, 20):
        for t in 100. * np.random.rand(100):
            assert abs(p.R(0., t, n0) - q(t, n0)) < 1e-4

