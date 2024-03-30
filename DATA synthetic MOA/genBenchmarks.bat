@Rem Java required.

@Rem Abrupt 
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 5 -a 20 -n 1 -t 0.5) -p 50000 -w 1) -f Abrupt_HP_5.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 1 -t 0.5) -p 50000 -w 1) -f Abrupt_HP_10.arff -m 100000"

@Rem Gradual 
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 5 -a 20 -n 1 -t 0.5) -p 50000 -w 20000) -f Gradual_HP_5.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 1 -t 0.5) -p 50000 -w 20000) -f Gradual_HP_10.arff -m 100000"

@Rem Recurring 
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20 -n 1) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -i 20 -k 5 -a 20 -n 1) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -k 5 -a 20 -n 1) -d (generators.HyperplaneGenerator -i 10 -k 5 -a 20 -n 1) -p 25000 -w 1) -p 25000 -w 1) -p 25000 -w 1) -f Recurring_HP_5.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20 -n 1) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 1) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -k 10 -a 20 -n 1) -d (generators.HyperplaneGenerator -i 10 -k 10 -a 20 -n 1) -p 25000 -w 1) -p 25000 -w 1) -p 25000 -w 1) -f Recurring_HP_10.arff -m 100000"

@Rem Incremental
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (generators.HyperplaneGenerator -i 20 -k 5 -a 20  -n 1 -t 0.01 -s 10) -f Incremental_HP_5.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (generators.HyperplaneGenerator -i 20 -k 10 -a 20  -n 1 -t 0.01 -s 10) -f Incremental_HP_10.arff -m 100000"
