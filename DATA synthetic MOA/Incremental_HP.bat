WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator –k 2 -a 20 -t 0.01 -s 10) -f incremental_HP.arff -m 100000