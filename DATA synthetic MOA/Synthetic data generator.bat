ABRUPT 
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 1) -p 50000 -w 1) -f Abrupt_HP_10_1.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 3) -p 50000 -w 1) -f Abrupt_HP_10_3.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 5) -p 50000 -w 1) -f Abrupt_HP_10_5.arff -m 100000"

java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 1) -p 50000 -w 1) -f Abrupt_HP_20_1.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 3) -p 50000 -w 1) -f Abrupt_HP_20_3.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 5) -p 50000 -w 1) -f Abrupt_HP_20_5.arff -m 100000"


GRADUAL 
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 1 -t 0.01) -p 50000 -w 20000) -f Gradual_HP_10_1.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 3 -t 0.01) -p 50000 -w 20000) -f Gradual_HP_10_3.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 5 -t 0.01) -p 50000 -w 20000) -f Gradual_HP_10_5.arff -m 100000"

java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 1 -t 0.01) -p 50000 -w 20000) -f Gradual_HP_20_1.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 3 -t 0.01) -p 50000 -w 20000) -f Gradual_HP_20_3.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 5 -t 0.01) -p 50000 -w 20000) -f Gradual_HP_20_5.arff -m 100000"


RECURRING 
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20 -n 1) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 1) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -k 10 -a 20 -n 1) -d (generators.HyperplaneGenerator -i 10 -k 10 -a 20 -n 1) -p 25000 -w 1) -p 25000 -w 1) -p 25000 -w 1) -f Recurring_HP_10_1.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20 -n 3) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 3) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -k 10 -a 20 -n 3) -d (generators.HyperplaneGenerator -i 10 -k 10 -a 20 -n 3) -p 25000 -w 1) -p 25000 -w 1) -p 25000 -w 1) -f Recurring_HP_10_3.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20 -n 5) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 5) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -k 10 -a 20 -n 5) -d (generators.HyperplaneGenerator -i 10 -k 10 -a 20 -n 5) -p 25000 -w 1) -p 25000 -w 1) -p 25000 -w 1) -f Recurring_HP_10_5.arff -m 100000"

java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20 -n 1) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 1) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -k 20 -a 20 -n 1) -d (generators.HyperplaneGenerator -i 10 -k 20 -a 20 -n 1) -p 25000 -w 1) -p 25000 -w 1) -p 25000 -w 1) -f Recurring_HP_20_1.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20 -n 3) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 3) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -k 20 -a 20 -n 3) -d (generators.HyperplaneGenerator -i 10 -k 20 -a 20 -n 3) -p 25000 -w 1) -p 25000 -w 1) -p 25000 -w 1) -f Recurring_HP_20_3.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20 -n 5) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 5) -d (ConceptDriftStream -s (generators.HyperplaneGenerator -k 20 -a 20 -n 5) -d (generators.HyperplaneGenerator -i 10 -k 20 -a 20 -n 5) -p 25000 -w 1) -p 25000 -w 1) -p 25000 -w 1) -f Recurring_HP_20_5.arff -m 100000"


INCREMENTAL 
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 1 -t 0.01) -p 50000 -w 100000) -f Incremental_HP_10_1.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 3 -t 0.01) -p 50000 -w 100000) -f Incremental_HP_10_3.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 10 -a 20 -n 5 -t 0.01) -p 50000 -w 100000) -f Incremental_HP_10_5.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 1 -t 0.01) -p 50000 -w 100000) -f Incremental_HP_20_1.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 3 -t 0.01) -p 50000 -w 100000) -f Incremental_HP_20_3.arff -m 100000"
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.HyperplaneGenerator -a 20) -d (generators.HyperplaneGenerator -i 20 -k 20 -a 20 -n 5 -t 0.01) -p 50000 -w 100000) -f Incremental_HP_20_5.arff -m 100000"
