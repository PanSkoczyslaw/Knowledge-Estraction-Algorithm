import json
import streamlit as st
from PIL import Image



with open("nn_recipe.json", "r") as data:
    nn = json.load(data)

with open("lp_from_nn.json", "r") as data:
    lp = json.load(data)


def nn_params(neural_network):
    nn_params = {"atoms":{"sum": 0, "input layer": 0, "hidden layer": 0, "output layer": 0, "recurrent layer": 0}, \
        "connections between neurons":{"from inp to hid": 0, "from hid to out": 0, "recurrent ": 0}, \
        "connections with weight bigger than amin": {"from inp to hid": 0, "from hid to out": 0, "recurrent ": 0}, "factors":{}}
    nn_params["atoms"]["sum"] = len((neural_network)["nn"]["inpLayer"])+len((neural_network)["nn"]["hidLayer"])+len((neural_network)["nn"]["outLayer"])
    nn_params["atoms"]["input layer"] = len((neural_network)["nn"]["inpLayer"]) # atomy na inputLayer
    nn_params["atoms"]["hidden layer"] = len((neural_network)["nn"]["hidLayer"]) # atomy na hiddenLayer
    nn_params["atoms"]["output layer"] = len((neural_network)["nn"]["outLayer"]) # atomy na outputLayer
    nn_params["connections between neurons"]["from inp to hid"] = len(neural_network["nn"]["inpToHidConnections"]) # połączenia inp2Hid
    nn_params["connections between neurons"]["from hid to out"] = len(neural_network["nn"]["hidToOutConnections"]) # połączenia hid2Out
    for i in neural_network["nn"]["inpToHidConnections"]:
        if abs(i["weight"]) >= neural_network["nnFactors"]["amin"]:
            nn_params["connections with weight bigger than amin"]["from inp to hid"] += 1 # duże wagi połączeń inp2Hid
    for i in neural_network["nn"]["hidToOutConnections"]:
        if abs(i["weight"]) >= neural_network["nnFactors"]["amin"]:
            nn_params["connections with weight bigger than amin"]["from hid to out"] += 1 # duże wagi połączeń hid2Out
    if "recLayer" in (neural_network)["nn"]:
        nn_params["atoms"]["sum"] += len((neural_network)["nn"]["recLayer"])
        nn_params["atoms"]["recurrent layer"] = len((neural_network)["nn"]["recLayer"]) # atomy w recLayer
        nn_params["connections between neurons"]["recurrent "] = len(neural_network["nn"]["recConnections"]) # połączenie rec
        for i in neural_network["nn"]["recConnections"]:
            if abs(i["weight"]) >= neural_network["nnFactors"]["amin"]:
                nn_params["connections with weight bigger than amin"]["recurrent "] += 1 # duże wagi połączeń rec
    nn_params["factors"] = neural_network["nnFactors"]
    return nn_params



def lp_params(logic_program):
    nnParams = {"clauses": {"amount": 0, "with positive atoms only": 0, "with negative atoms only": 0, "with positive and negative atoms": 0, "with head with H label": 0, \
        "atoms": {"amount": 0, "positive": 0, "negative": 0, "with H label": 0, "positive with H label": 0, "negative with H label": 0}, "different atoms": {"amount": 0, "positive": 0, "negative": 0, "with H label": 0, "positive with H label": 0, "negative with H label": 0}}, "facts": 0, "assumptions": 0}
    for i in range(len(logic_program)):
        if logic_program[i]["tag"] == "Cl": # klauzule
            nnParams["clauses"]["amount"] += 1
            if len(logic_program[i]["clPAtoms"]) == 0: # klauzule onlyNegAtoms
                nnParams["clauses"]["with negative atoms only"] +=1 
            elif len(logic_program[i]["clNAtoms"]) == 0: # klauzule onlyPosAtoms
                nnParams["clauses"]["with positive atoms only"] +=1 
            else: #klauzule mixAtoms
                nnParams["clauses"]["with positive and negative atoms"] +=1
            nnParams["clauses"]["atoms"]["amount"] += len(logic_program[i]["clPAtoms"]) + len(logic_program[i]["clNAtoms"]) # suma klauzul
            if logic_program[i]["clHead"]["label"] == "h": # klauzule z hHead
                nnParams["clauses"]["with head with H label"] += 1
            
            atoms = []
            for j in logic_program[i]["clPAtoms"]: # atomy pozytywne
                nnParams["clauses"]["atoms"]["positive"] +=1
                if j["label"] == "h": # atomy pozytywne z h
                        nnParams["clauses"]["atoms"]["with H label"] += 1
                        nnParams["clauses"]["atoms"]["positive with H label"] += 1
                if not j in atoms: # atomy pozytywne bez powtórzeń
                    atoms.append(j)
                    nnParams["clauses"]["different atoms"]["positive"] += 1
                    if j["label"] == "h": #atomy pozytywne bez powtórzeń z h
                        nnParams["clauses"]["different atoms"]["with H label"] += 1
                        nnParams["clauses"]["different atoms"]["positive with H label"] += 1

            for j in logic_program[i]["clNAtoms"]: # atomy negatywne
                nnParams["clauses"]["atoms"]["negative"] +=1
                if j["label"] == "h": # atomy negatywne z h
                        nnParams["clauses"]["atoms"]["with H label"] += 1
                        nnParams["clauses"]["atoms"]["negative with H label"] += 1
                if not j in atoms: # atomy negatywne bez powtórzeń
                    atoms.append(j)
                    nnParams["clauses"]["different atoms"]["negative"] += 1
                    if j["label"] == "h": # atomy negatywne bez powtórzeń z h
                        nnParams["clauses"]["different atoms"]["with H label"] += 1
                        nnParams["clauses"]["different atoms"]["negative with H label"] += 1
            nnParams["clauses"]["different atoms"]["amount"] = len(atoms) # atomy bez powtórzeń
        elif logic_program[i]["tag"] == "Fact": # fakty
            nnParams["facts"] += 1
        elif logic_program[i]["tag"] == "Assumption": # assumpitons
            nnParams["assumptions"] += 1
    return nnParams
            




st.title('NN&LP JSONs Analysis')

st.markdown("## 1. Structure of the logic program\n \
### 1.1. Horn Clause\n \
** Horn clause ** is an expression of the form:\n \
<center>aj ← ak , . . . , an;</center>\n \
where:", unsafe_allow_html = True)

st.markdown(" * aj, ak , . . . , an are atomic formulas\n \
* head(hi) = aj\n \
* body(hi) = {ak , . . . , an}")

st.markdown("### 1.2. Logic Program\n \
** Logic program ** is a non-empty set of Horn clauses:\n \
<center>P = {h1, . . . , hn};</center> \n \
where:", unsafe_allow_html = True)

st.markdown(" * hi for 1 ≤ i ≤ n are a Horn clauses\n ")
lp_example = Image.open("lp_example.png")
st.markdown("#### 1.2.1 Example of a logic program: ")
st.image(lp_example)

st.markdown("### 1.3. Logic program structure in JSON format")
st.write(lp)

st.markdown("## 2. Summmary of logic program parameters")
st.write(lp_params(lp))

st.markdown("## 3. Structure of neural network")
nn_schema = Image.open("nn_schema.png")
st.image(nn_schema)

st.markdown("Our neural network has:\n \
* 3 layers of neurons with recurrent connections from output to input layer\n \
* hidden and output layer neurons have bipolar sigmoid activation function\n \
* input layer neurons have identity activation function\n \ ")

st.markdown("### 3.1 Neural netowrk structure in JSON format")
st.write(nn)


st.markdown("## 4. Summary of neural network parameters")
st.write(nn_params(nn))
