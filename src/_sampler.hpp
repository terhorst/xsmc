#include <random>
#include <algorithm>
#include <vector>

struct sampler_key  {
    int u, v, delta, Xb;
    sampler_key() {}
    sampler_key(int u, int v, int delta, int Xb) : u(u), v(v), delta(delta), Xb(Xb) {}

	// operator== is required to compare keys in case of hash collision
	bool operator==(const sampler_key &s) const
	{
		return u == s.u && v == s.v && delta == s.delta && Xb == s.Xb;
	}
};

// specialized hash function for unordered_map keys
struct hash_fn
{
	std::size_t operator() (const sampler_key &s) const
	{
		std::size_t h1 = std::hash<int>()(s.u);
		std::size_t h2 = std::hash<int>()(s.v);
		std::size_t h3 = std::hash<int>()(s.delta);
		std::size_t h4 = std::hash<int>()(s.Xb);
		return h1 ^ h2 ^ h3 ^ h4;
	}
};

template <typename U>
double rexp(U &gen) {
    return std::exponential_distribution<>(1.)(gen);
}

template <typename T, typename U>
void shuffle(std::vector<T> &v, U &gen)
{
    std::shuffle(v.begin(), v.end(), gen);
}
