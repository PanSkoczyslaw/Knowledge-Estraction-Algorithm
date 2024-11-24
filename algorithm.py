import json
from urllib.request import urlopen, Request
# import matplotlib.pyplot as plt
import numpy as np
from typing import List, Iterable
import itertools
import src.neural_network.activations
import src.toolbox as tb

class Tristate:

    def __init__(self, value=None):
        if any(value is v for v in (True, False, None)):
            self.value = value
        else:
            raise ValueError("Tristate value must be True, False, or None")

    def __eq__(self, other):
        return (self.value is other.value if isinstance(other, Tristate)
                else self.value is other)

    def __ne__(self, other):
        return not self == other

    def __bool__(self):
        raise TypeError("Tristate object may not be used as a Boolean")

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return "Tristate(%s)" % self.value

    def isFalse(self):
        return self.value is False

    def isTrue(self):
        return self.value is True

    def isNone(self):
        return self.value is None

    def __invert__(self):
        if self.isNone():
            return Tristate(None)
        return Tristate(not self.value)


def tristate_all(tristates: [Tristate]) -> Tristate:
    """
    Implementation of function all() for tristate logic (logic conjunction).

    :param tristates: list of tristates (list of Tristate)
    :return: Tristate

    """
    if all(map(lambda x: x.isTrue(), tristates)):
        return Tristate(True)
    if any(map(lambda x: x.isFalse(), tristates)):
        return Tristate(False)
    return Tristate(None)


def tristate_any(tristates: [Tristate]) -> Tristate:
    """
    Implementation of function any() for tristate logic ( logic alternative).

    :param tristates: list of tristates (list of Tristate)
    :return: Tristate

    """
    if any(map(lambda x: x.isTrue(), tristates)):
        return Tristate(True)
    if any(map(lambda x: x.isNone(), tristates)):
        return Tristate(None)
    return Tristate(False)


def tristate_implication(antecedent: Tristate, consequent: Tristate) -> Tristate:
    if antecedent.isFalse():
        return Tristate(True)
    if consequent.isTrue():
        return Tristate(True)
    if antecedent.isTrue() and consequent.isFalse():
        return Tristate(False)
    return Tristate(None)
############################################################################

class Factors:

    def __init__(self,
                 beta: float,
                 ahln: float,
                 r: float,
                 bias: float,
                 w: float,
                 amin: float):
        self.beta = beta
        self.ahln = ahln
        self.bias = bias
        self.amin = amin
        self.r = r
        self.w = w

    @staticmethod
    def from_dict(d: dict):
        return Factors(**d)

    def to_dict(self):
        return {'beta': self.beta,
                'ahln': self.ahln,
                'r': self.r,
                'bias': self.bias,
                'w': self.w,
                'amin': self.amin}


class Atom:

    def __init__(self, idx: int, label: str = 'n'):
        self.idx = idx
        self.label = label
        self.negated = 'n' in label

    def __eq__(self, other):
        if isinstance(other, Atom):
            return self.idx == other.idx
        raise TypeError(f"Atom can be compared only to another atom, not with {type(other)}")

    def __hash__(self):
        return self.idx

    def __repr__(self):
        return f"Atom idx: {self.idx}, label: {self.label}"

    def to_dict(self):
        return {"idx": self.idx, "label": self.label}

    def evaluate(self, positive, negative) -> Tristate:
        if self.negated:
            if self in negative:
                return Tristate(True)
            if self in positive:
                return Tristate(False)
            return Tristate(None)
        else:
            if self in positive:
                return Tristate(True)
            if self in positive:
                return Tristate(False)
            return Tristate(None)


class Clause:

    def __init__(self, head: Atom, positive: [Atom], negative: [Atom], tag: str = ''):
        self.head = head
        self.positive = positive
        self.negative = negative
        self.tag = tag

    @staticmethod
    def from_dict(d: dict):
        return Clause(head=Atom(**d['clHead']),
                      positive=[Atom(**spec) for spec in d['clPAtoms']],
                      negative=[Atom(**spec) for spec in d['clNAtoms']],
                      tag=d['tag'])

    def to_dict(self):
        return {"tag": self.tag,
                "clHead": self.head.to_dict(),
                "clPAtoms": [atom.to_dict() for atom in self.positive],
                "clNAtoms": [atom.to_dict() for atom in self.negative]}

    def calculate(self, positive: [Atom], negative: [Atom]) -> Tristate:
        assert not contradiction(positive, negative)

        antecedent_positive = [atom.evaluate(positive, negative) for atom in self.positive]
        antecedent_negative = [atom.evaluate(positive, negative) for atom in self.negative]

        return tristate_all(antecedent_positive + antecedent_negative)


class LogicProgram:

    def __init__(self, facts: [Clause], assumptions: [Clause], clauses: [Clause]):
        self.facts = facts
        self.assumptions = assumptions
        self.clauses = clauses
        self.all_clauses = self.facts + self.assumptions + self.clauses

    @staticmethod
    def from_dict(d: dict):
        return LogicProgram(facts=[Clause.from_dict(spec) for spec in d['facts']],
                            assumptions=[Clause.from_dict(spec) for spec in d['assumptions']],
                            clauses=[Clause.from_dict(spec) for spec in d['clauses']])

    @staticmethod
    def from_json(json_string: str):
        lp_dict = json.loads(json_string)
        if 'lp' in lp_dict:
            lp_dict = lp_dict['lp']
        return LogicProgram.from_dict(lp_dict)

    @staticmethod
    def from_file(fp: str):
        with open(fp, 'r') as file:
            json_string = file.read()
        return LogicProgram.from_json(json_string)

    def to_dict(self) -> dict:
        return {"facts": [cl.to_dict() for cl in self.facts],
                "assumptions": [cl.to_dict() for cl in self.assumptions],
                "clauses": [cl.to_dict() for cl in self.clauses]}

    def add_clause(self, clause: Clause):
        if not clause.positive or clause.negative:
            self.facts.append(clause)
        else:
            self.clauses.append(clause)
        self.all_clauses.append(clause)

    def to_json(self) -> str:
        return json.loads(self.to_dict())

    def tp_single_iteration(self, positive: [Atom], negative: [Atom]):
        new_positive = [clause.head for clause in self.all_clauses if clause.calculate(positive, negative).isTrue()]
        new_negative = [clause.head for clause in self.all_clauses if clause.calculate(positive, negative).isFalse()]
        print(new_positive, new_negative)
        return new_positive, new_negative

    def tp(self):
        new_positive, new_negative = [], []

        while True:
            positive, negative = new_positive, new_negative
            new_positive, new_negative = self.tp_single_iteration(positive, negative)
            if (set(positive) == set(new_positive)) and (set(negative) == set(new_negative)):
                break

        return new_positive, new_negative


def get(f, phrase, url='http://64.225.103.216:10100/api/'):
    """
    Opens url using Request library

    :param f: to which function you want to connect (Str)
    :param phrase: request phrase (Str)
    :param url: url of server (Str)
    :return: response (Str)

    """
    request = Request(url+f, phrase.encode("utf-8"))
    # print(request.get_full_url())
    response = urlopen(request)
    html = response.read()
    response.close()
    return html.decode("utf-8")


def get_lp_from_nn(order_inp: [str], order_out: [str], amin: float, io_pairs: [tuple]) -> dict:

    request_dict = {"orderInp": order_inp,
                    "orderOut": order_out,
                    "amin": amin,
                    "ioPairs": io_pairs}
    request_json = json.dumps(request_dict)
    response = get('nn2lp', request_json)
    return json.loads(response)




def get_nn_recipe(logic_program: LogicProgram,
                  abductive_goal: Clause,
                  factors: Factors) -> dict:
    """
    Get a Neural Network Recipe from API.

    :param logic_program: logic program (src.logic.LogicProgram)
    :param abductive_goal: abductive goal for abductive process (src.logic.Clause)
    :param factors: factors for neural network (src.logic.Factors)
    :return: recipe for neural network (dict)

    """
    request_dict = {"lp": logic_program.to_dict(),
                    "abductive_goal": abductive_goal.to_dict(),
                    "factors": factors.to_dict()}

    request_json = json.dumps(request_dict)
    return json.loads(get('lp2nn', request_json))

with open('example.json', 'r') as json_file:
    json_content = json.load(json_file)


lp = LogicProgram.from_dict(json_content['lp'])
ag = Clause.from_dict(json_content['abductive_goal'])
factors = Factors.from_dict(json_content['factors'])

print(get_nn_recipe(lp, ag, factors))


################################################################################
#### NEURAL NETWORKS ###########################################################
################################################################################


def mean_squarred_error(y, output):
    return (y - output) ** 2


def d_mean_squarred_error(y, output):
    return 2 * (output - y)


def valuation(x, a_min, binary=False):
    # print("binary:", binary)
    if (type(x) == list) or (type(x) == np.ndarray):
        return [valuation(elem, a_min, binary) for elem in x]
    else:
        if x >= a_min:
            return 1

        else:
            if binary or x <= (-1 * a_min):
                return -1
            return 0


def act_f(f: str) -> callable:
    """
    :param f: string name of function
    :return: fuction of taht name from activations.py
    """
    if f == "idem":
        raw_f = src.neural_network.activations.idem
    elif f == "const":
        raw_f = src.neural_network.activations.const
    elif f == "sigm" or f == "tanh":
        raw_f = src.neural_network.activations.sigm
    else:
        raise ValueError(f"There is no function named {f} in activation function list.")

    return raw_f


def to_binary(matrix: np.ndarray) -> np.ndarray:
    """
    Creates boolean mask of given array.

    :param matrix: numpy.ndarray
    :return: numpy.ndarray

    """
    return np.vectorize(bool)(matrix)


class LayerInfo:

    def __init__(self, specification: List[dict]):
        self.specification = specification

        self.label = [neuron['label'] for neuron in specification]
        self.f = [neuron['activFunc'] for neuron in specification]
        self.bias = [neuron['bias'] for neuron in specification]
        self.idx = [neuron['idx'] for neuron in specification]

        self.len = len(specification)


def flatten_rec_layer(connections: [dict]) -> [dict]:
    flat_connections = []
    rec_layer_froms = dict()
    rec_layer_tos = dict()

    for connection in connections:

        if connection['toNeuron'].startswith('rec'):
            # Create mapping {"out3": "recA2"} // {"outLayer": "addRecLayer"}
            rec_layer_froms[connection['fromNeuron']] = connection['toNeuron']

        elif connection['fromNeuron'].startswith('rec'):
            # Create mapping {"recA2": "inp3"} // {"addRecLayer": "inpLayer"}
            rec_layer_tos[connection['fromNeuron']] = connection['toNeuron']

        else:
            flat_connections.append(connection)

    for from_neuron, rec_neuron in rec_layer_froms.items():
        to_neuron = rec_layer_tos[rec_neuron]
        flat_connections.append({'fromNeuron': from_neuron, 'toNeuron': to_neuron, 'weight': 1.0})

    return flat_connections


def set_weights(connections: [dict], prev_layer: LayerInfo, next_layer: LayerInfo) -> np.ndarray:
    """
    Sets weights based on connections dicts and layers.

    :param connections: [dict]
    :param prev_layer: LayerInfo
    :param next_layer: LayerInfo
    :return: numpy.ndarray

    """
    weights = np.zeros((next_layer.len, prev_layer.len))

    prev_dict = dict(zip(prev_layer.idx, [i for i in range(prev_layer.len)]))
    next_dict = dict(zip(next_layer.idx, [i for i in range(next_layer.len)]))

    for connection in connections:
        weights[next_dict[connection['toNeuron']]][prev_dict[connection['fromNeuron']]] = connection['weight']

    return weights


def function_vector(fs: [callable]) -> callable:
    def try_(try_f: callable, *try_args, **try_kwargs):
        try:
            return try_f(*try_args, **try_kwargs)
        except Exception as e:
            print("Values when error was raised:")
            print("args:", try_args)
            print("kwargs:", try_kwargs)
            raise e

    def vectorized(vector):
        if len(fs) != len(vector):
            raise ValueError('Length of vector must be the same as length of function vector')
        return [try_(f, x) for f, x in zip(fs, vector)]

    return vectorized


def get_model(values: [int], order: [str]) -> dict:
    model = {"positive": [], "negative": []}
    for value, label in zip(values, order):
        if value == 1:
            model["positive"].append(label)
        elif value == -1:
            model["negative"].append(label)
    return model


class NeuralNetwork3L:

    def __init__(self, architecture: dict, factors: src.logic.Factors, eta=-0.01):

        self.comments = []

        self.architecture = architecture

        self.inp_layer_spec = LayerInfo(architecture['inpLayer'])
        self.hid_layer_spec = LayerInfo(architecture['hidLayer'])
        self.out_layer_spec = LayerInfo(architecture['outLayer'])

        self.i2h_connections = set_weights(architecture['inpToHidConnections'],
                                           self.inp_layer_spec, self.hid_layer_spec)
        self.h2o_connections = set_weights(architecture['hidToOutConnections'],
                                           self.hid_layer_spec, self.out_layer_spec)
        self.o2i_connections = set_weights(flatten_rec_layer(architecture['recConnections']),
                                           self.out_layer_spec, self.inp_layer_spec)

        self.factors = factors
        self.eta = eta

        # TODO: Make sure that truth neuron can't switch weight
        self.i2h_b = to_binary(self.i2h_connections)
        self.h2o_b = to_binary(self.h2o_connections)
        self.o2i_b = to_binary(self.o2i_connections)

        self.inp_layer_aggregated = np.zeros(self.inp_layer_spec.len)
        self.hid_layer_aggregated = np.zeros(self.hid_layer_spec.len)
        self.out_layer_aggregated = np.zeros(self.out_layer_spec.len)

        self.inp_layer_calculated = np.zeros(self.inp_layer_spec.len)
        self.hid_layer_calculated = np.zeros(self.hid_layer_spec.len)
        self.out_layer_calculated = np.zeros(self.out_layer_spec.len)

        self.inp_layer_activation = function_vector([act_f(f) for f in self.inp_layer_spec.f])
        self.hid_layer_activation = function_vector([act_f(f) for f in self.hid_layer_spec.f])
        self.out_layer_activation = function_vector([act_f(f) for f in self.out_layer_spec.f])

        self.inp_layer_activation_derivation = function_vector(
            [act_f(f).__getattribute__('d') for f in self.inp_layer_spec.f])
        self.hid_layer_activation_derivation = function_vector(
            [act_f(f).__getattribute__('d') for f in self.hid_layer_spec.f])
        self.out_layer_activation_derivation = function_vector(
            [act_f(f).__getattribute__('d') for f in self.out_layer_spec.f])

        self.errors = []

        self.output_error = None
        self.output_delta = None
        self.hidden_error = None
        self.hidden_delta = None

        self.shape = f"{self.inp_layer_spec.len}x{self.hid_layer_spec.len}x{self.out_layer_spec.len}"

    @staticmethod
    def from_dict(d: dict):
        return NeuralNetwork3L(architecture=d['nn'], factors=src.logic.Factors(**d['nnFactors']))

    @staticmethod
    def from_json(j: str):
        return NeuralNetwork3L.from_dict(json.loads(j))

    @staticmethod
    def from_file(j: str):
        with open(j, 'r') as json_file:
            return NeuralNetwork3L.from_dict(json.load(json_file))

    @staticmethod
    def from_lp(lp: src.logic.LogicProgram, ag: src.logic.Clause, factors: src.logic.Factors):

        nn_recipe = src.connect.get_nn_recipe(lp, ag, factors)
        return NeuralNetwork3L.from_dict(nn_recipe)

    @staticmethod
    def from_dropped(fp: str):
        with open(fp, 'r') as json_file:
            dropped = json.load(json_file)

        nn = NeuralNetwork3L.from_dict(dropped)
        nn.i2h_connections = dropped['i2h_connections']
        nn.h2o_connections = dropped['h2o_connections']
        nn.o2i_connections = dropped['o2i_connections']

        nn.i2h_b = to_binary(nn.i2h_connections)
        nn.h2o_b = to_binary(nn.h2o_connections)
        nn.o2i_b = to_binary(nn.o2i_connections)

        nn.comments = dropped['comments']

        return nn

    def _pack(self) -> dict:
        return {'architecture': self.architecture,
                'i2h_connections': self.i2h_connections.tolist(),
                'h2o_connections': self.h2o_connections.tolist(),
                'o2i_connections': self.o2i_connections.tolist(),
                'factors': self.factors.to_dict(),
                'comments': self.comments}

    def drop(self, fp: str):
        d = self._pack()
        with open(fp, 'w') as json_file:
            json.dump(d, json_file)

    def to_lp(self):

        return src.connect.get_lp_from_nn(
            order_inp=[label for label in self.inp_layer_spec.label],
            order_out=[label for label in self.out_layer_spec.label],
            # order_inp=[{"idx": idx, "label": label} for idx, label in zip(self.inp_layer_spec.idx, self.inp_layer_spec.label)],
            # order_out=[{"idx": idx, "label": label} for idx, label in zip(self.out_layer_spec.idx, self.out_layer_spec.label)],
            amin=self.factors.amin,
            io_pairs=self.get_io_pairs())

    def calculate_input(self, x: np.ndarray = None):
        pass

    def forward(self, x: np.ndarray):
        """
        Implementation of feed forward.

        Function that calculates output of Neural Network by given input.
        :param: input_vector: one-dimensional numpy.ndarray or other iterable
                one dimensional object.
        """

        self.inp_layer_aggregated = x - self.inp_layer_spec.bias
        self.inp_layer_calculated = np.array(self.inp_layer_activation(self.inp_layer_aggregated))

        self.hid_layer_aggregated = self.i2h_connections.dot(self.inp_layer_calculated) - self.hid_layer_spec.bias
        self.hid_layer_calculated = np.array(self.hid_layer_activation(self.hid_layer_aggregated))

        self.out_layer_aggregated = self.h2o_connections.dot(self.hid_layer_calculated) - self.out_layer_spec.bias
        self.out_layer_calculated = np.array(self.out_layer_activation(self.out_layer_aggregated))

        return self.out_layer_calculated

    def backprop(self, y: np.ndarray, eta: float):
        """
        Implementation of Backpropagation algorithm.
        Function that calculates the error on the output layer, propagates it
        through network and modifies weights according using delta rule.

        Every weight in network is modified using following equation:

        ΔW_ho = η * (y-o) * f'(v) * h   , where
        η     = learning constant, usually very small, around 0.01 (self.eta)
        (y-o) = squared error derivative
        y     = values of the output layer (self.XXX_layer_calculated)
        o     = values expected on the output (x)
        f'    = derivative of the activation function (self.XXX_layer_activation_derivation)
        v     = aggregated values on next layer

        More about Backpropagation agorithm:
        > https://en.wikipedia.org/wiki/Backpropagation

        More about Delta Rule:
        > https://en.wikipedia.org/wiki/Delta_rule

        """
        self.error = mean_squarred_error(y, self.out_layer_calculated)
        self.output_error = d_mean_squarred_error(y, self.out_layer_calculated)  # error in output
        self.output_delta = self.output_error * self.out_layer_activation_derivation(self.out_layer_aggregated)

        # z2 error: how much our hidden layer weights contribute to output error
        self.hidden_error = self.output_delta.dot(self.h2o_connections)

        # applying derivative of sigmoid to z2 error
        self.hidden_delta = self.hidden_error * self.hid_layer_activation_derivation(self.hid_layer_aggregated)

        # adjusting first set (input -> hidden) weights
        l2h_weights_delta = np.outer(self.inp_layer_calculated.T, self.hidden_delta).T * eta
        self.i2h_connections += l2h_weights_delta * self.i2h_b

        # adjusting second set (hidden -> output) weights
        h2o_weights_delta = np.outer(self.hid_layer_calculated.T, self.output_delta).T * eta
        self.h2o_connections += h2o_weights_delta * self.h2o_b

        # print(l2h_weights_delta)
        # print(h2o_weights_delta)

        return sum(self.error) / len(self.error)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, on_stabilised: bool = False,
              stop_when: callable = lambda e: e == 0, vis=False):

        examples_n = len(x)
        self.errors = []

        fig = None

        for epoch in range(epochs):
            for i, (x_, y_) in enumerate(zip(x, y)):
                if on_stabilised:
                    self.stabilize(x_)
                else:
                    self.forward(x_)
                avg_error = self.backprop(y_, self.eta)
                # self.update_weights(new_i2h_connections, new_h2o_connections, eta)
                self.errors.append(avg_error)
                # print(f'Epoch {epoch + 1}/{epochs} | Example {i + 1}/{examples_n} | Error: {avg_error}')

                if vis:
                    if fig is not None:
                        fig.clear()
                    fig = self.draw(fig=fig)

                if stop_when(avg_error):
                    return 0

    def stabilize(self, x: Iterable = None, binary=False):
        """
        All inputs to -1

        One neuron always return True.

        Network is considered stable when input and output (True - None - False)

        """

        # print(x)

        if x is None:
            x = np.array([-1 for _ in range(self.inp_layer_spec.len)])
        if len(x) != self.inp_layer_spec.len:
            raise ValueError(f"x must have length {self.inp_layer_spec.len}, has {len(x)} instead.")

        last_output_vector = []

        output_vector = self.forward(x)
        tp_iteration = 0

        # print("Tp Operator iteration:", tp_iteration)
        # print("Output vector:", valuation(output_vector, self.factors.amin, binary))
        # print("Model", get_model(valuation(output_vector, self.factors.amin, binary), self.out_layer_spec.label))

        while valuation(last_output_vector, self.factors.amin, binary) != valuation(output_vector, self.factors.amin,
                                                                                    binary):

            # TODO: Find where is bug
            last_output_vector = output_vector
            input_vector = self.o2i_connections.dot(self.out_layer_calculated)
            output_vector = self.forward(input_vector)

            tp_iteration += 1

            # TODO this has to change according to logic program
            if tp_iteration >= 1000:
                print('not stabilised')
                self.comments.append('not stabilised')
                break

            # print("Tp Operator iteration:", tp_iteration)
            # print("Output vector:", valuation(output_vector, self.factors.amin, binary))
            # print("Model", get_model(valuation(output_vector, self.factors.amin, binary), self.out_layer_spec.label))

        return output_vector

    def get_io_pairs(self):
        inputs = tb.all_combinations(self.inp_layer_spec.len)
        io_pairs = []

        for x in inputs:
            y = self.forward(np.array(list(x)))
            io_pairs.append((list(x), y.tolist()))

        return io_pairs

    def set_true(self, true: [str]) -> np.ndarray:

        a = np.array([-1 for _ in range(self.inp_layer_spec.len)])
        for true_neuron in true:
            a[self.inp_layer_spec.label.index(true_neuron)] = 1

        return a


with open('example.json', 'r') as json_file:
    json_content = json.load(json_file)

lp = LogicProgram.from_dict(json_content['lp'])
ag = Clause.from_dict(json_content['abductive_goal'])
factors = Factors.from_dict(json_content['factors'])

f = open('example.json')
data = json.load(f)

recipe_dict_sup = get_nn_recipe(lp, ag, factors)
print(recipe_dict_sup)
network = NeuralNetwork3L.from_dict(recipe_dict_sup)

def dist(vector1, vector2):
    counts = 0
    for i in range(len(vector1)):
        if vector1[i] == vector2[i]:
            counts = counts
        else:
            counts += 1
    return abs(counts)


def dist_for_CL(vector1, vector2):
    counts = 0
    for i in range(len(vector1)-1):
        if vector1[i][0] == vector2[i][0]:
            counts = counts
        else:
            counts += 1
    return abs(counts)

def sum_vec(vector):
    counts = 0
    for i in vector:
        counts += vector[i]
    return counts

# Funkcja dystans i sum_vec. Dystans jest potrzebny do działania głównej pętli, ale z kolei tego sum_vec za specjalnie nie używam

def flipping_inf(vector, factor, counter):
    new_inf = vector.copy()
    if new_inf[-1 - factor] == -1:
        new_inf[-1 - factor] = 1
    if counter >= 1:
        if new_inf[-1] == -1:
            new_inf[-1] = 1
        for subfactor in range(counter):
            if new_inf[-1 - subfactor] == -1:
                new_inf[-1 - subfactor] = 1

    return new_inf


def flipping_supp(vector, factor, counter):
    new_supp = vector.copy()
    if new_supp[0 + factor] == 1:
        new_supp[0 + factor] = -1
    if counter >= 1:
        if new_supp[0] == 1:
            new_supp[0] = -1
        for subfactor in range(counter):
            if new_supp[0 + subfactor] == 1:
                new_supp[0 + subfactor] = -1
    return new_supp

# Funkcje do przerzucania jedynek/minusjedynek w wektorze w poszukiwaniu aktywacji outputu. U Garceza działanie jest opisane w stópce na stronie 134


def SSPR2R(vector1, neuron, nn_inp, rules):
    network.forward(vector1)
    if network.forward(vector1)[neuron] > 0.1:
        factor = 0
        rules.append(vector1)
        while factor < nn_inp:
            new_vector = vector1.copy()
            if new_vector[-1 - factor] == -1:
                new_vector[-1 - factor] = 1
                rules.append(new_vector)
            factor +=1

        applicable = True
        return applicable

    else:
        applicable = False
        return applicable

# Funkcje Search Space Pruning Rule 1 i 2. Są opisane u Garceza na stronie 131. W poprzednich wersjach pobierały one po 2 wektory
# (co można zobaczyć bo ich nie usunąłem), ale stwierdziłem że w formie z pobieraniem jednego wektora jest lepiej.
# SSPR2R sprawdza czy dany neuron w warstwie wyjściowej jest aktywowany w pobranym wektorze i jeśli tak, to znaczy że każdy inny wektor
# większy o 1 (czyli taki który ma jedną jedynkę w sobie więcej) też będzie aktywował. Wszystkie wektory zostają dodane do zbioru reguł
# i funkcja zwraca zmienną applicable jako True. Od tego momentu przestają być generowane nowe wektory typu inf.


def SSPR1R(vector1, neuron, rules):

    network.forward(vector1)
    if network.forward(vector1)[neuron] < 0.1:
        applicable = True
        return applicable

    else:
        applicable = False
        rules.append(vector1)
        return applicable

def ComplementaryLiterals(rules):
    new_rules = rules.copy()
    garry = []
    # NIE WIEM XD
    indicator = 0
    factor = 0
    operator = 0
    for i in range(len(new_rules)):
        new_rules[i] = [[el] for el in new_rules[i]]
    for i in new_rules:
        for j in i:
            # if rules[i][j] == facts[j]:
            # new_rules.append(rules[i][j])
            j.append(network.architecture['inpLayer'][indicator]['label'])
            indicator += 1
        indicator = 0
    new_rules_derived = new_rules.copy()
    while operator != len(new_rules):
        for i in range(len(new_rules)):
            # if new_rules[i] != new_rules[-1]:

            if dist_for_CL(new_rules[operator], new_rules[i]) == 1:
                while factor != len(new_rules[0]):
                    for j in range(i):
                        # print(new_rules[operator][factor][0], new_rules[i][j][0])
                        if new_rules[operator][factor][0] != new_rules[i][j][0]:
                            if new_rules[i] not in new_rules_derived:
                                new_rules_derived.append(new_rules[i])
                            garry = new_rules[i].copy()
                                # TU SIE JUZ DZIEJA RZECZY NIESTWORZONE
                            new_rules_derived[i].remove(new_rules_derived[i][j])
                            if garry not in new_rules_derived:
                                new_rules_derived.append(garry)
                    factor += 1
            else:
                pass

        operator +=1
    return new_rules_derived
# SSPR1R działa o tyle inaczej od SSPR2R że ona zwraca zmienną applicable jako True dopiero gdy neuron z warstwy wyjściowej z pobranego
# wektora nie zostaje aktywowany. Od tego momentu nie są generowane kolejne wektory typu supp. Z kolei jeśli neuron jest aktywowany to
# wszystkie kolejne wektory zostają dodane do zbioru reguł

def MofN(rules, n_lenght, supp):
    for i in range(n_lenght, -1, -1):
       comb = itertools.combinations(supp, i)
       for j in comb:
           if j not in rules:
               break
           else:
                pass

       MofN_set = i
    return MofN_set


# print(dist(i_inf, i_supp))
# print(sum_vec(i_supp))
# print(sum_vec(i_inf))


# tab = np.array([1, 1, 1, 1, 1, 1])




tab_supp = np.array([1, 1, 1, 1, 1,1])
tab_inf = np.array([-1, -1, -1, -1, -1,-1])
i_inf = tab_inf.copy()
i_supp = tab_supp.copy()
new_inf = tab_inf.copy()
new_supp = tab_supp.copy()
nn_input = len(tab_supp)
neuron = 0

# Parametry startowe. nn_input to zmienna określająca ile jest neuronów w wartswie wejściowej,
# używana w pętli i w paru innych miejscach jako zmienna pomocnicza
# neuron, czyli po prostu od którego neuronu z warstwy wyjściowej zaczynamy iterowanie. Jest ich o jeden mniej niż liczy sobie zmienna nn_input

factor_inf = 0
factor_supp = 0
rules = []
rules.append(tab_supp)
supp_applicable = False
inf_applicable = False
counter = 0
facts = []
negatives = []
new_rules = []
permission = None
# print(network.forward(tab_supp)[neuron])

nn_rules = []
for i in range(nn_input):
    nn_rules.append([])

print(nn_rules)

new_rules = nn_rules.copy()

# print(network.architecture['inpLayer'][1]['label'])
# print(network.forward(tab_inf))
# print(network.forward(tab_supp))
# print(network.forward(np.array([1,-1,1,1,1,1]))[0], 'aktywacja neuronu')
# print(network.forward(np.array([1,1,-1,1,1,1]))[0], 'aktywacja neuronu')
# print(network.forward(np.array([1,1,1,-1,1,1]))[0], 'aktywacja neuronu')
# print(network.forward(np.array([1,1,1,1,-1,-1]))[0], 'aktywacja neuronu')

# factor_inf i supp mają na początku zero i używam ich w funkcjach flipping do tego aby z każdą iteracją zmieniać kolejny element wektora
# counter jest kolejną zmienną której używam we flippach, aby w momencie w którym iteracja dojdzie do początku/końca wektora, zaczynać znowu
# flippowanie po wektorze, ale już nie od początku, a od kolejnego z brzegu elementu. Początkowy element jest już zmieniony na stałe.
# Czyli jak zaczynamy np od [-1,-1,-1] i najpierw funkcja flippuje [-1,-1,1], potem [-1,1,-1], potem [1,-1,-1] to wtedy zaczyna od początku
# od kolejnego z brzegu elementu, a początkowy jest już na stałe zmieniony czyli [-1,1,1], potem [1,-1,1] itd.
# Lista facts jest do zbierania neuronów które zawsze zwracają dodatnią aktywacje niezależnie od podanego inputu, a negatives które zawsze
# dają ujemną aktywacje niezależnie od inputu. Takie naurony są pomijane w algorytmie, o czym decyduje zmienna permission.
# Cały algorytm jest opisany u Garceza na 134 stronie

while neuron != nn_input -1:
    # Rozpoczynamy ekstrakcje reguł dla każdego neuronu z warstwy wyjściowej
    counter = 0
    factor_inf = 0
    factor_supp = 0
    # rules = []
    supp_applicable = False
    inf_applicable = False
    # new_rules = []
    i_inf = tab_inf.copy()
    i_supp = tab_supp.copy()
    permission = None
    new_inf = tab_inf.copy()
    new_supp = tab_supp.copy()
    network.forward(tab_inf)
    if network.forward(tab_inf)[neuron] > 0.1:
        facts.append(neuron)
        permission = None


    elif network.forward(tab_supp)[neuron] < 0.1:
        negatives.append(neuron)
        permission = None
    else:
        permission = True
        nn_rules[neuron].append(tab_supp)
    if permission:
        print(neuron, 'numer atomu')
        while (dist(tab_inf, i_inf) <= (nn_input/2) and inf_applicable == False) or (dist(tab_supp, i_supp) <= (nn_input/2 + nn_input % 2) and supp_applicable == False):
            print(dist(tab_inf, i_inf))
            print(dist(tab_supp, i_supp))
            # print(tab_inf)
            if not inf_applicable and dist(new_inf, i_inf) <= 2:
                new_inf = flipping_inf(tab_inf, factor_inf, counter)
                factor_inf += 1
                if counter >= 1:
                    # TU TRZEBA ZROBIC COS Z INF_INPUT == 1 ABY ON SIE ZMIENIAL ZA KAZDYM RAZEM JAK INF ZROBI PELNE OKRAZENIE
                    i_inf = new_inf.copy()
                else:
                    i_inf = i_inf

            #     Tak jak opisałem wcześniej, w momencie w którym SSPR2R nie jest aplikowalne to generuje kolejne new_inf
            else:
                i_inf = i_inf
                new_inf = new_inf

            # print(tab_inf)
            # print(network.forward(new_inf))
            # print(new_inf)
            # print(tab_inf)
            if not inf_applicable:
                inf_applicable = SSPR2R(new_inf, neuron, nn_input, nn_rules[neuron])
            # apply the simplification complementary literals... (step 4)
            if not supp_applicable and dist(new_supp, i_supp) <= 2:
                i_supp = new_supp.copy()
                new_supp = flipping_supp(tab_supp, factor_supp, counter)
                factor_supp += 1
            #     Analogicznie jak wyżej, tylko dla new_supp
            else:
                i_supp = i_supp
                new_supp = new_supp
            # print(network.forward(new_supp))
            if not supp_applicable:
                supp_applicable = SSPR1R(new_supp, neuron, nn_rules[neuron])

            # step 8
            # factor_inf += 1
            # factor_supp += 1
            if factor_inf >= nn_input or factor_supp >= nn_input:
                counter += 1
                factor_supp = counter
                factor_inf = counter
            #     To tutaj właśnie flippowanie po wektorze zaczyna się od kolejnego z brzegu elementu


            # i_inf = new_inf.copy()
            # print(factor_inf)
            # print(supp_permission)
            # print(inf_permission)
            print(nn_rules[neuron], 'rules')

            print(new_inf)

            print(new_supp)
            print(supp_applicable, 'supp appl')
            print(inf_applicable, 'inf appl')
            print(dist(new_supp, i_supp), 'dystans supp')
            print(dist(new_inf, i_inf), 'dystans inf')
            print(network.forward(new_inf)[neuron], 'aktywacja neuronu inf')
            print(network.forward(new_supp)[neuron], 'aktywacja neuronu supp')
            print(factor_inf, 'inf factor')
            print(neuron, 'neuron')
            ComplementaryLiterals(nn_rules[neuron])
            new_rules[neuron] = ComplementaryLiterals(nn_rules[neuron])
            print(new_rules)
            if counter == nn_input:
                break
    #
    # SSPR1R zwraca True w drugiej iteracji, więc dodane zostają tylko dwa wektory do rules, czyli [1,1,1,1,1,1] i [-1,1,1,1,1,1].
    # SSPR2R zwraca True w czwartej iteracji więc do zbioru reguł zostaje dodany wektor [-1,1,-1,-1,-1,-1] i wszystkie o 1 większe które
    # mają drugi element ustawiony na 1. Dalej nic już się nie zmienia.

    neuron +=1
    permission = None

print(new_rules[0], 'rulesy dla wszystkich atomów warstwy wyjściowej')
print(new_rules[1])
print(new_rules[2])
print(facts, 'fakty')
print(negatives, 'negativesy')

# lista = [[1,1,1,1,1,1], [1,1,1,1,1,1], [1,-1,1,-1,-1,-1]]
# for i in range(len(lista)):
#     lista[i] = [[el] for el in lista[i]]
# print(lista)
# for i in lista:
#         for j in i:
#             j.append('A')
# # for i in lista:
# #     for j in range(len(i)):
# #        print(i[j][0])
# for i in range(len(lista)):
#     for j in range(i):
#         print(lista[i][j][0])
# vector1= [[1], [1], [1], [-1], [1], [1]]
# vector2 = [[1], [-1], [1], [-1], [-1], [-1]]
# # for i in range(len(vector2)):
# #
# #     print(vector2[i][0])
# counts = 0
# indicator = 0
# while indicator != len(vector1):
#     for i in range(len(vector1)):
#         print(vector1[indicator][0], vector2[i][0])
#         if vector1[indicator][0] == vector2[i][0]:
#
#             counts = counts
#         else:
#             counts += 1
#     indicator += 1
#     print(counts)
# # print(lista)