Kod zawiera dwie funkcje w pliku jsnons.py:
  - nn_params
  - lp_params
  
 Obydwie jako argumentu przyjmują ściężkę pliku json, w którym zapisana jest:
   - sieć neuronowa
   - program logiczny
 
 Na wyjściu otrzymujemy json z podsumowaniem parametrów sieci neuronowej lub programu logicznego.
 
 Wyjście z lp_params zawiera wartości paramtrów:
 >             {"clauses": {
 >                          "amount": 0, 
 >                         "onlyPos": 0,
 >                          "onlyNeg": 0,
 >                         "mix": 0,
 >                          "headWithH": 0,
 >                          "atoms": {
 >                                    "sum": 0,
 >                                    "pos": 0,
 >                                    "neg": 0,
 >                                    "withH": 0,
 >                                    "posWithH": 0,
 >                                    "negWithH": 0
 >                                    },
 >                         "difAtoms": {
 >                                    "sum": 0,
 >                                    "pos": 0,
 >                                    "neg": 0,
 >                                    "withH": 0,
 >                                    "posWithH": 0,
 >                                    "negWithH": 0
 >                                    }
 >                          },
 >              "facts": 0,
 >              "assumptions": 0
 >              }
              
              
Wyjście z nn_params zawiera wartości parametrów:
 >              {"atoms":{
 >                      "sum": 0,
 >                      "inp": 0,
 >                      "hid": 0,
 >                      "out": 0,
 >                      "rec": 0
 >                      },
 >                "connections":{
 >                       "inp2Hid": 0,
 >                       "hid2Out": 0,
 >                       "rec": 0
 >                      },
 >                "bigWeights":{
 >                        "inp2Hid": 0,
 >                        "hid2Out": 0,
 >                        "rec": 0
 >                        },
 >                 "factors":{
 >                        "beta": 0,
 >                        "ahln": 0,
 >                        "r": 0,
 >                        "bias": 0,
 >                        "w": 0,
 >                        "amin": 0
 >                        }
 >                  }
