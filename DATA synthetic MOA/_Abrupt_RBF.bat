java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomRBFGenerator -a 20) -d (generators.RandomRBFGeneratorDrift -i 10 -s 2.0 -a 20) -p 50000 -w 1) -f Abrupt_RBF_10.arff -m 100000"

java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomRBFGenerator -a 20) -d (generators.RandomRBFGeneratorDrift -i 20 -s 2.0 -a 20) -p 50000 -w 1) -f Abrupt_RBF_20.arff -m 100000"

java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomRBFGenerator -a 20) -d (generators.RandomRBFGeneratorDrift -i 30 -s 2.0 -a 20) -p 50000 -w 1) -f Abrupt_RBF_30.arff -m 100000"

java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomRBFGenerator -a 20) -d (generators.RandomRBFGeneratorDrift -i 40 -s 2.0 -a 20) -p 50000 -w 1) -f Abrupt_RBF_40.arff -m 100000"

java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomRBFGenerator -a 20) -d (generators.RandomRBFGeneratorDrift -i 50 -s 2.0 -a 20) -p 50000 -w 1) -f Abrupt_RBF_50.arff -m 100000"
