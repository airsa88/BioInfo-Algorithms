
#@Author - Antony

# Script to read the input sequences and generate the Feature vectors by using C-T-D method for each property.
# BioSeqClass (V3.6) and Biostrings R packages are downloaded from BioConductor used for this features construction.
# Source for packages:
# “BioSeqClass,” Bioconductor. [Online]. Available: https://www.bioconductor.org/packages/release/bioc/html/BioSeqClass.html
# “Biostrings,” Bioconductor. [Online]. Available: https://www.bioconductor.org/packages/release/bioc/html/Biostrings.html
# Please ensure that you have the below dependent libraries 
# such as scatterplot3d,ipred, e1071, klaR, randomForest, class, tree,
# coin, combinat, digest, glue, htmltools, httpuv, IRanges, jsonlite, lava, magrittr, mime, modeltools, multcomp, mvtnorm, numDeriv, 
# prodlim, R6, Rcpp, S4Vectors, sandwich, sourcetools, SQUAREM, stringi, stringr, strucchange, TH.data, xtable, XVector, zlibbioc, 
# zoo, nnet, rpart, party, foreign, Biobase, utils, stats, grDevices
# installed before run this script.
#
#
  
  library(scatterplot3d)  
  library(BioSeqClass)
  
  # Features creation for the Training data by using C-T-D for five physicochemical properties. 
  # File path for training input data (positive and negative)
  tmpDir=file.path(path.package('BioSeqClass'), 'example')
  tmpTrainPosFile=file.path(tmpDir, 'RBP2780.faa')
  tmpTrainNegFile=file.path(tmpDir, 'Non-RBP2323.faa')
  
  # Read the input file sequence by sequence.
  library(Biostrings)
  tmp1 = readAAStringSet(tmpTrainPosFile)
  trainPosSequence = paste(tmp1)
  tmp2 = readAAStringSet(tmpTrainNegFile)
  trainNegSequence = paste(tmp2)
  
  # Create positive and negative sequence matrix
  posSeq=as.matrix(trainPosSequence)
  negSeq=as.matrix(trainNegSequence)
  trainSeqNew=c(posSeq,negSeq)
  
  # Length of total positive and negative samples.
  length(trainSeqNew)
  
  # Features construction using CTD for property: aaH - (hydrophobicity)
  hydroCTD = featureCTD(trainSeqNew, class=aaClass("aaH") )
    
  # Features construction using CTD for property: aaV - (normalized Van der Waals volume)
  vanderCTD = featureCTD(trainSeqNew, class=aaClass("aaV") )	
	
  # Features construction using CTD for property: aaF - (Charge and polarity of side chain)
  chargeCTD = featureCTD(trainSeqNew, class=aaClass("aaF") )	
	
  # Features construction using CTD for property: aaP - (Polarity)
  polarCTD = featureCTD(trainSeqNew, class=aaClass("aaP") )	
	
  # Features construction using CTD for property: aaZ - (Polarizability)
  polarizabilityCTD = featureCTD(trainSeqNew, class=aaClass("aaZ") )	
	 	
		
  # Features creation for the test datasets for three species by using C-T-D for five physicochemical properties. 
  
  # Organism 1 : H.sapiens
  # File path for testing data (positive and negative)
  tmpDir=file.path(path.package('BioSeqClass'), 'example')
  tmpTestPosFileHuman=file.path(tmpDir, 'Human-RBP967.faa')
  tmpTestNegFileHuman=file.path(tmpDir, 'Human-non-RBP588.faa')
  
  # Read the input file sequence by sequence.
  library(Biostrings)
  tmp1 = readAAStringSet(tmpTestPosFileHuman)
  testPosSeqHuman = paste(tmp1)
  tmp2 = readAAStringSet(tmpTestNegFileHuman)
  testNegSeqHuman = paste(tmp2)
  
  # Create positive and negative sequence matrix
  posSeq=as.matrix(testPosSeqHuman)
  negSeq=as.matrix(testNegSeqHuman)
  testSeqNewHuman=c(posSeq,negSeq)
  
  # Length of total positive and negative samples.
  length(testSeqNewHuman)
  
  # Features construction using CTD for property: aaH - (hydrophobicity)
  hydroCTDHuman = featureCTD(testSeqNewHuman, class=aaClass("aaH") )
    
  # Features construction using CTD for property: aaV - (normalized Van der Waals volume)
  vanderCTDHuman = featureCTD(testSeqNewHuman, class=aaClass("aaV") )	
	
  # Features construction using CTD for property: aaF - (Charge and polarity of side chain)
  chargeCTDHuman = featureCTD(testSeqNewHuman, class=aaClass("aaF") )	
	
  # Features construction using CTD for property: aaP - (Polarity)
  polarCTDHuman = featureCTD(testSeqNewHuman, class=aaClass("aaP") )	
	
  # Features construction using CTD for property: aaZ - (Polarizability)
  polarizabilityCTDHuman = featureCTD(testSeqNewHuman, class=aaClass("aaZ") )	
	
	
  # Organism 2 : S.cerevisiae 
  # File path for testing data (positive and negative)
  tmpDir=file.path(path.package('BioSeqClass'), 'example')
  tmpTestPosFileCere=file.path(tmpDir, 'Scerevisiae-RBP354.faa')
  tmpTestNegFileCere=file.path(tmpDir, 'Scerevisiae-non-RBP135.faa')
  
  # Read the input file sequence by sequence.
  library(Biostrings)
  tmp1 = readAAStringSet(tmpTestPosFileCere)
  testPosSeqCere = paste(tmp1)
  tmp2 = readAAStringSet(tmpTestNegFileCere)
  testNegSeqCere = paste(tmp2)
  
  # Create positive and negative sequence matrix
  posSeq=as.matrix(testPosSeqCere)
  negSeq=as.matrix(testNegSeqCere)
  testSeqNewCere=c(posSeq,negSeq)
  
  # Length of total positive and negative samples.
  length(testSeqNewCere)
  
  # Features construction using CTD for property: aaH - (hydrophobicity)
  hydroCTDCere = featureCTD(testSeqNewCere, class=aaClass("aaH") )
    
  # Features construction using CTD for property: aaV - (normalized Van der Waals volume)
  vanderCTDCere = featureCTD(testSeqNewCere, class=aaClass("aaV") )	
	
  # Features construction using CTD for property: aaF - (Charge and polarity of side chain)
  chargeCTDCere = featureCTD(testSeqNewCere, class=aaClass("aaF") )	
	
  # Features construction using CTD for property: aaP - (Polarity)
  polarCTDCere = featureCTD(testSeqNewCere, class=aaClass("aaP") )	
	
  # Features construction using CTD for property: aaZ - (Polarizability)
  polarizabilityCTDCere = featureCTD(testSeqNewCere, class=aaClass("aaZ") )	

  
  # Organism 3 : A.thaliana 
  # File path for testing data (positive and negative)
  tmpDir=file.path(path.package('BioSeqClass'), 'example')
  tmpTestPosFileThaliana=file.path(tmpDir, 'Athaliana-RBP456.faa')
  tmpTestNegFileThaliana=file.path(tmpDir, 'Athaliana-non-RBP37.faa')
  
  # Read the input file sequence by sequence.
  library(Biostrings)
  tmp1 = readAAStringSet(tmpTestPosFileThaliana)
  testPosSeqThaliana = paste(tmp1)
  tmp2 = readAAStringSet(tmpTestNegFileCereThaliana)
  testNegSeqThaliana = paste(tmp2)
  
  # Create positive and negative sequence matrix
  posSeq=as.matrix(testPosSeqThaliana)
  negSeq=as.matrix(testNegSeqThaliana)
  testSeqNewThaliana=c(posSeq,negSeq)
  
  # Length of total positive and negative samples.
  length(testSeqNewThaliana)
  
  # Features construction using CTD for property: aaH - (hydrophobicity)
  hydroCTDThaliana = featureCTD(testSeqNewThaliana, class=aaClass("aaH") )
    
  # Features construction using CTD for property: aaV - (normalized Van der Waals volume)
  vanderCTDThaliana = featureCTD(testSeqNewThaliana, class=aaClass("aaV") )	
	
  # Features construction using CTD for property: aaF - (Charge and polarity of side chain)
  chargeCTDThaliana = featureCTD(testSeqNewThaliana, class=aaClass("aaF") )	
	
  # Features construction using CTD for property: aaP - (Polarity)
  polarCTDThaliana = featureCTD(testSeqNewThaliana, class=aaClass("aaP") )	
	
  # Features construction using CTD for property: aaZ - (Polarizability)
  polarizabilityCTDThaliana = featureCTD(testSeqNewThaliana, class=aaClass("aaZ") )
  
  # To download the CTD feature output as text file to local system.
  write.table(hydroCTDThaliana, "c:/hydroCTDThaliana.txt", sep="\t")

  
  
		