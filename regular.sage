def factorize(x, max_prime):
    factors = dict(factor(x))
    max_prime = max_prime or max(factors.keys())
    return [factors.get(i, 0) for i in primes(max_prime + 1)]

def odd_limit(n):
    # range() doesn't return Sage's divisible integers
    return {Integer(y)/x
        for x in range(1, n, 2)
        for y in range(x+2, n+1, 2)}

def farey_limit(n):
    # range() doesn't return Sage's divisible integers
    return {Integer(y)/x
        for x in range(1, n)
        for y in range(x+1, n+1)}

def make_limit(ratios):
    """
    Factorize rationals to a consistent vector size
    """
    max_prime = max(p for q in ratios for p, e in factor(q))
    vectors = [factorize(q, max_prime) for q in ratios]
    return prime_limit(max_prime), vectors

def prime_limit(p):
    return [log(i)/log(2) for i in primes(p+1)]

def temper_out(*ratios):
    limit = max(p for q in ratios for p, e in factor(q))
    unison_vectors = matrix([factorize(q, limit) for q in ratios])
    return unison_vectors.right_kernel().matrix().hermite_form()


#
# TE things
#

def TE_complexity(plimit, mapping, prec=53):
    """
    M can be a matrix of a list of lists
    but plimit must be a list
    """
    W = TE_weighting(plimit, prec=prec)
    M = matrix(RealField(prec), mapping) if prec else matrix(mapping)
    return size_of_matrix(M, W*W / len(plimit))

def TE_badness(plimit, mapping, Ek=0, prec=53):
    """
    Cangwu badness with a TE metric
    """
    Mp = matrix(mapping) * Cangwu_transformation(plimit, Ek, prec=prec)
    return size_of_matrix(Mp)

def Cangwu_badness(plimit, mapping, Ek=0, prec=53):
    """
    TE_badness with a different formula
    """
    G = Cangwu_metric(plimit, Ek, prec=prec)
    return size_of_matrix(matrix(mapping), G)

def Quadratic_badness(plimit, mapping, Ek=0, prec=53):
    """
    Another Cangwu formula
    """
    badness = TE_badness(plimit, mapping, prec=prec)
    complexity = TE_complexity(plimit, mapping, prec=prec)
    raw = badness**2 + (Ek * complexity)**2
    return sqrt(raw / (1 + Ek**2), prec)

def Cangwu_transformation(plimit, Ek=0, prec=53):
    """
    Matrix to transform a mapping matrix into "Cangwu badness" space.
    """
    epsilon = Ek / sqrt(1 + Ek**2, prec)
    W = TE_weighting(plimit, prec=prec)
    J = matrix([1]*len(plimit))
    trans = W - (1 - epsilon) * W*J.transpose() / len(plimit) * J
    return matrix(trans / sqrt(len(plimit), prec))

def Cangwu_metric(plimit, Ek=0, prec=53):
    W = TE_weighting(plimit, prec=prec)
    J = matrix([1]*len(plimit))
    metric = W*W - (W * J.transpose()*J * W) / len(plimit) / (1 + Ek**2)
    return metric / len(plimit)

def TE_error(plimit, mapping, prec=53):
    badness = TE_badness(plimit, mapping, prec=prec)
    complexity = TE_complexity(plimit, mapping, prec=prec)
    return badness / complexity

def TE_generators(plimit, mapping):
    M = matrix(mapping)
    J = matrix([1]*len(plimit))
    W = TE_weighting(plimit)
    return J*(M*W).pseudoinverse()

def TE_weighting(plimit, prec=53):
    """
    plimit must be a list not a matrix
    """
    rank = len(plimit)
    W = matrix(RealField(prec) if prec else SR, rank, rank)
    for i, p in enumerate(plimit):
        W[i, i] = 1/p
    return W

def size_of_matrix(M, metric=None):
    if metric is None:
        gram = M * M.transpose()
    else:
        gram = M * metric * M.transpose()
    return sqrt(gram.determinant())

def badness_equivalence_proof(rank, dimension):
    """
    Try to show if two different badness calculations are identical
    for a given rank and dimension.
    Should return True but might not even for a valid equivalence
    """
    assert rank in ZZ
    assert dimension in ZZ
    assert 0 < rank <= dimension
    mapping = [
            var(['m' + str(i) + str(j) for j in range(dimension)],
                domain='integer')
            for i in range(rank)]
    plimit = var(['p' + str(i) for i in range(dimension)], domain='real')
    var('Ek')
    cangwu = Cangwu_badness(plimit, mapping, Ek, prec=None)
    quadratic = Quadratic_badness(plimit, mapping, Ek, prec=None)
    bad2 = Cangwu_badness(plimit, mapping, prec=None)**2

    def simplify_result(result):
        # Rearrange these to be Ek**2 * complexity**2
        # which gets rid of square roots and is simpler
        comp2 = result**2 * (Ek**2 + 1) - bad2
        # Solving for Ek**2 removes a variable
        # and puts the expression in a standard form
        [solution] = solve(comp2 == 1, Ek**2)
        assert solution.lhs() == Ek**2
        return solution.rhs()

    lhs = simplify_result(cangwu)
    rhs = simplify_result(quadratic)
    return bool(lhs == rhs)

def badness_equivalence_simple_proof(rank, dimension):
    """
    Try to show if two different simplifications of Cangwu badness
    are equivalent.
    """
    assert rank in ZZ
    assert dimension in ZZ
    assert 0 < rank <= dimension
    # M is a ready-weighted mapping
    M = matrix(rank, dimension,
        var(['m' + str(i) for i in range(rank * dimension)]))
    var('Ek2')  # Ek**2
    MJ = matrix(list(map(mean, M)))
    MM = M * M.transpose() / dimension
    MJJM = MJ.transpose() * MJ
    comp2 = MM.determinant()
    bad2 = (MM - MJJM).determinant()
    cangwu2 = expand(MM - MJJM / (1 + Ek2)).determinant() * (1 + Ek2)
    [solution] = (cangwu2 - bad2 == 1).solve(Ek2)
    assert solution.lhs() == Ek2
    lhs = solution.rhs()
    rhs = 1 / comp2
    return bool(lhs == rhs)


#
# Minimax things
#

def odd_limit_minimax(limit, mapping):
    plimit, vectors = make_limit(odd_limit(limit))
    solver = MinimaxByMapping(plimit, vectors, mapping, True)
    return solver.optimal_generators()

def odd_limit_hermite_minimax(limit, mapping):
    return odd_limit_minimax(limit, matrix(ZZ, mapping).echelon_form())

def temper_out_minimax(*ratios):
    limit = max(p for q in ratios for p, e in factor(q))
    unison_vectors = [factorize(q, limit) for q in ratios]
    plimit, consonances = make_limit(odd_limit(limit))
    solver = MinimaxByUnisonVectors(plimit, consonances, unison_vectors, True)
    return solver.tuning_map()


class MinimaxOptimizer(object):
    def set_standard_constraints(self, octave_equivalent=False):
        p = self.linear_program
        error = p.new_variable(real=True, nonnegative=True)['e']
        for vec in self.consonances:
            # Sage's linear function doesn't work in a constraint,
            # so cast log(p)/log(2) to a real
            target = sum(m*RR(p) for m, p in zip(vec, self.plimit))
            approx = sum(m*h for m, h in zip(vec, self.harmonics))
            p.add_constraint(approx - target <= error)
            p.add_constraint(target - approx <= error)
        if octave_equivalent:
            p.add_constraint(self.plimit[0] == self.harmonics[0])
        p.set_objective(-error)


class MinimaxByMapping(MinimaxOptimizer):
    def __init__(self, plimit, vectors, mapping, octave_equivalent=False):
        self.plimit = plimit
        self.linear_program = MixedIntegerLinearProgram()
        self.mapping = matrix(mapping)
        self.generators = self.linear_program.new_variable(
                real=True, nonnegative=False)
        self.harmonics = [
            sum(m*p for (m, p) in zip(pmap, self.generators))
            for pmap in zip(*mapping)]
        self.consonances = vectors
        self.set_standard_constraints(octave_equivalent)

    def optimal_generators(self):
        self.linear_program.solve()
        generators = self.linear_program.get_values(self.generators)
        return matrix([g for i, g in sorted(generators.items())])

    def tuning_map(self):
        return self.optimal_generators() * self.mapping


class MinimaxByUnisonVectors(MinimaxOptimizer):
    def __init__(self, plimit, consonances, unison_vectors,
                                    octave_equivalent=False):
        self.plimit = plimit
        self.linear_program = MixedIntegerLinearProgram()
        self.harmonics = self.linear_program.new_variable(real=True)
        self.consonances = consonances
        for vec in unison_vectors:
            unison = sum(uv*h for uv, h in zip(vec, self.harmonics))
            self.linear_program.add_constraint(unison == 0)
        self.set_standard_constraints(octave_equivalent)

    def tuning_map(self):
        self.linear_program.solve()
        harmonics = self.linear_program.get_values(self.harmonics)
        return matrix([g for i, g in sorted(harmonics.items())])


class TOPByMapping(MinimaxByMapping):
    def __init__(self, plimit, mapping):
        self.plimit = [1]*len(plimit)
        self.linear_program = MixedIntegerLinearProgram()
        self.mapping = matrix(mapping)
        weighted_mapping = self.mapping * TE_weighting(plimit)
        self.generators = self.linear_program.new_variable(
                real=True, nonnegative=False)
        self.harmonics = [
            sum(m*p for (m, p) in zip(pmap, self.generators))
            for pmap in zip(*weighted_mapping)]
        self.consonances = matrix(ZZ, len(plimit), len(plimit), 1)
        self.set_standard_constraints(False)


def kernel(mapping):
    return matrix(mapping).right_kernel().matrix().LLL(delta=0.3)
