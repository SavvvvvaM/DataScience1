library(shiny)
library(readr)
library(tm)
#dyn.load('/Library/Java/JavaVirtualMachines/jdk1.8.0_101.jdk/Contents/Home/jre/lib/server/libjvm.dylib')
#library(qdap)
#I'm not using qdap cause it's a pain in a neck
library(caret)
library(tidyr)
library(plyr)
library(dplyr)
#library(RWeka)
library(MASS)
library(SnowballC)
library(stringr)
library(mlbench)
library(pROC)
library(SDMTools)
library(wordcloud)
library(shinycssloaders)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  
  observeEvent(input$showSVM, {
    showModal(modalDialog(
      title = "What is SVM?",
      HTML(
        "&emsp;&emsp; Support Vector Machine (SVM) is an algorithm that plots each data item as a point in n-dimensional space (where n is number of features) with the value of each feature being the value of a particular coordinate. 
        Then we perform classification by finding the hyper-plane that differentiates the two classes very well. Example:
        <br>
        <img src=\"svm.png\">
        <br>
        &emsp;&emsp;Support Vector Machine is a hyper-plane/line that best segregates the two classes. 
        While A,B,C are all apropriate, B is considred the best.
        "), 
      easyClose = TRUE,
      footer = NULL
      ))
  })
  
  
  observeEvent(input$showLDA, {
    showModal(modalDialog(
      title = "What is LDA?",
      HTML(
        "&emsp;&emsp; Linear Discriminant Analysis (LDA) is a classification method that creates a linear combination of features using statistical properties of your data calculated for each class. 
        The goal of the LDA is to project your n-dim feature space onto a smaller k-dim subspace while maintaining the class-discriminatory information. 
        <br>
        <img src=\"lda.png\">
        <br>
        "), 
      easyClose = TRUE,
      footer = NULL
      ))
  })
  
  
  observeEvent(input$showKNN, {
    showModal(modalDialog(
      title = "What is KNN?",
      HTML(
        "&emsp;&emsp; K Nearest Neighbors is an algorithm that classifies new points based on their distance to already existing points.
        If among K nearest points to X the majority are spam - then X will be spam too. Example:
        <br>
        <img src=\"knn.png\">
        <br>
        &emsp;&emsp;for K = 3, green is classified as red, for K = 5, green is classified as blue.
        "), 
      easyClose = TRUE,
      footer = NULL
    ))
  })
  
  observeEvent(input$showTREE, {
    showModal(modalDialog(
      title = "What is a Decision Tree?",
      HTML(
        "&emsp;&emsp; Decision Tree is an algorithm that uses conditional control statements to classify the elements; 
        it creates a tree-like graph of decisions and their possible consequences based on chance of event outcomes. Example:
        <br><br>
        <img src=\"tree.jpg\">
        <br><br>
        &emsp;&emsp;Every node represents a “test” and every branch represents an outcome.
        "), 
      easyClose = TRUE,
      footer = NULL
      ))
  })
  
  observeEvent(input$showNNET, {
    showModal(modalDialog(
      title = "What is a Neural Network?",
      HTML(
        "&emsp;&emsp; Artificial Neural Networks is a system (algorithm) vaguely inspired by the biological neural networks = brains. 
        The algorithm learns to complete a task by creating a collection of connected units or nodes called artificial neurons - all on its own. 
        <br><br>
        &emsp;&emsp; Neural network is organized into fead-forward layers of nodes (data moves in 1 direction). 
        The node receives data from the layer beneath it, multiplies this data by certain weights, and if the acquired result is higher than a certain threshold, passes the processed data to the nodes in the layer above it. 
        <br><br>        
        <img src=\"nnet.png\" height = \"295\" width = \"504\" >
        <br><br>
        &emsp;&emsp; In the process of training the model, all weights and thresholds are originally random. 
        Training dataset is fed into the input later, and the result is taken from the output layer. 
        In the training process, the weights and thresholds are continually adjusted until the model yields proper results.
        "), 
      easyClose = TRUE,
      footer = NULL
    ))
  })
  
  observeEvent(input$showCUT, {
    showModal(modalDialog(
      title = "What are cutoffs?",
      HTML("&emsp;&emsp; For every message my model calculates the probability of this message being spam or not. The probability cutoff would be a threshold probability value that determines whether the message is classified as spam.<br> 
           &emsp;&emsp; By changing this cutoff we can change the performances of our models."
           ),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  
  
  
  
  
  cleaned_data <- reactive({
    validate(
      need(input$file1 != "", "Please upload your data set")
    )
    
    #browser()
    #inFile = input$file1
    full = read.csv(input$file1$datapath)
    colnames(full) = c("type", "sms")
    full$type = factor(full$type) #switch from string to factor for simpler access/manipulation
    
    corp = VCorpus( VectorSource(full$sms) ) # make corpus
    
    cleaned_corp = corp #edit a copy of the corp
    #create the toSpace content transformer
    toSpace = content_transformer(function(x, pattern) {return (gsub(pattern, " ", x))})
    cleaned_corp = tm_map(cleaned_corp, toSpace, "!")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\"")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "#")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "$")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "%")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "&")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\(") # a pain in the neck
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\)")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\*")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\+")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\-")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\.")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\"")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "/")
    cleaned_corp = tm_map(cleaned_corp, toSpace, ":")
    cleaned_corp = tm_map(cleaned_corp, toSpace, ";")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "<")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "=")
    cleaned_corp = tm_map(cleaned_corp, toSpace, ">")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\?")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "@")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\[")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\]")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\^")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\_")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\{")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\|")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\}")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "~")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\.")
    
    
    
    
    #APPLY DEFAULT TRANSFORMATIONS
    #1.
    #remove numbers
    cleaned_corp = tm_map(cleaned_corp, removeNumbers)
    
    wcaps_corp = cleaned_corp #save it just in case beofre getting rid of caps
    #3.
    #remove caps letters.
    #cleaned_corp = tm_map(cleaned_corp, tolower)
    #NOTE: ha! it looks like tolower, being a non standard transformation, transforms messages NOT into text files. dunno why. fixed by making it a content_transformer()
    cleaned_corp = tm_map(cleaned_corp, content_transformer(tolower))
    #Later we will include number of caps, number of stop words, etc as parameters.
    #4.
    #Get rid of stop words
    cleaned_corp = tm_map(cleaned_corp, removeWords, stopwords())
    #5.
    #Get rid of tons of space bars
    cleaned_corp = tm_map(cleaned_corp, stripWhitespace)
    #Remove Punctuation
    cleaned_corp = tm_map(cleaned_corp, removePunctuation)
    #6.
    #Stem the words
    #Converting verbs to their root word. Example: "fighting" or "figths" becomes "fight".
    #NOTE: it doesn't turn fought to fight
    #Downloaded the first stemming package I found - "SnowballC". There is also "textstem", but I don't know.
    #wordStem(c("fighting", "fought", "fight", "fights", "fighter"))
    cleaned_corp = tm_map(cleaned_corp, stemDocument)
    
    
    
    
    
    #print("done")
    #})
    #output$calculateA <- renderPrint({
    dtm_text = DocumentTermMatrix(cleaned_corp)
    #findFreqTerms(dtm_text, 100, 30000) #interesting. I bet free, get, text, stop, now, reply, today are primarily from spam messages
    
    #make it a matrix just to take a look
    m_text = as.matrix(dtm_text) 
    
    #Very high sparsity, gotta remove some of that. I am using a rather low cutoff value here - 1% - reason is that later I want to combine multiple words into one variable using PCA.
    dtm_text_sparsed = removeSparseTerms(dtm_text, 0.99)
    m_text = as.matrix(dtm_text_sparsed) #make it into a matrix to extract the words and to take a look
    best_words = colnames(m_text)
    
    #DTM with most frequent words only
    dtm_freq_text = dtm_text[, best_words]
    df_words = as.data.frame(as.matrix(dtm_freq_text))
    #Make a data frame
    df_words = bind_cols(full[,1] %>% as.character() %>% as.data.frame(),df_words)
    names(df_words)[names(df_words) == "."] = "type"
    
    
    
    
    temp = full$sms %>% as.character() %>% as.data.frame()
    #names(temp) = NULL
    #1. add number of characters
    x = data.frame(apply(temp, 2, nchar))
    colnames(x) = c("nchars")
    df_words = bind_cols(x, df_words)
    #2. add number of block capitols
    #note: GOODNESS GRACIOUS, this thing didn't work for an hour
    numOfCaps = function(str){ #function that counts caps
      ldply(str_match_all(str,"[A-Z]"), length)  
    }
    x = data.frame(apply(temp, 2, numOfCaps))
    colnames(x) = c("ncaps")
    df_words = bind_cols(x, df_words)
    
    #3. add number of symbols
    numOfSymbols = function(str){ #function that counts symbols
      ldply(str_match_all(str,"[!-@]"), length)  
    }
    x = data.frame(apply(temp, 2, numOfSymbols))
    colnames(x) = c("nsymbs")
    df_words = bind_cols(x, df_words)
    #print("done")
    #})
    #output$calculateB <- renderPrint({
    
    test = df_words
    #Pre Process
    preProc = preProcess(test, method = c("center", "scale") )
    test1 = predict(preProc, test)
    #Create new pre-process, account for pca and boxcox.
    preProc2 = preProcess(test, method=c("BoxCox", "center", "scale", "pca"), thresh = 0.95)
    test2 = predict(preProc2, test)
    
    list(test = test)
  })
  

  
  output$plotKNN1 <- renderPrint({
    complete_test = cleaned_data()$test
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit4$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit4$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    c_test = predict(preProc, c_test)
    
    testPred_4 = predict(fit4, c_test, type = "prob")
    best_cutoff = 0.08012821
    my_cutoff = 0.1
    preds2 = ifelse(testPred_4$spam > best_cutoff, "spam", "ham")
    confusionMatrix(preds2, c_test$type)
  })
  
  output$plotKNN2 <- renderPrint({
    complete_test = cleaned_data()$test
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit4$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit4$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    c_test = predict(preProc, c_test)
    
    testPred_4 = predict(fit4, c_test, type = "prob")
    best_cutoff = 0.08012821
    my_cutoff = 0.1
    preds2 = ifelse(testPred_4$spam > my_cutoff, "spam", "ham")
    confusionMatrix(preds2, c_test$type)
  })
  
  output$plotSVM1 <- renderPrint({
    complete_test = cleaned_data()$test
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit7$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit7$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    c_test = predict(preProc, c_test)
    
    testPred_7 = predict(fit7, c_test, type = "prob")
    best_cutoff = 0.1788879
    my_cutoff = 0.29
    preds7 = ifelse(testPred_7$spam > best_cutoff, "spam", "ham")
    confusionMatrix(preds7, c_test$type)
  })
  
  output$plotSVM2 <- renderPrint({
    complete_test = cleaned_data()$test
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit7$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit7$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    c_test = predict(preProc, c_test)
    
    testPred_7 = predict(fit7, c_test, type = "prob")
    best_cutoff = 0.1788879
    my_cutoff = 0.29
    preds7 = ifelse(testPred_7$spam > my_cutoff, "spam", "ham")
    confusionMatrix(preds7, c_test$type)
  })
  
  output$plotLDA1 <- renderPrint({
    complete_test = cleaned_data()$test
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit9$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit9$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    c_test = predict(preProc, c_test)
    
    testPred_9 = predict(fit9, c_test, type = "prob")
    best_cutoff =  0.0001217802
    my_cutoff = 0.05
    preds9 = ifelse(testPred_9$spam > best_cutoff, "spam", "ham")
    confusionMatrix(preds9, c_test$type)
  })
  
  output$plotLDA2 <- renderPrint({
    complete_test = cleaned_data()$test
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit9$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit9$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    c_test = predict(preProc, c_test)
    
    testPred_9 = predict(fit9, c_test, type = "prob")
    best_cutoff =  0.0001217802
    my_cutoff = 0.05
    preds9 = ifelse(testPred_9$spam > my_cutoff, "spam", "ham")
    confusionMatrix(preds9, c_test$type)
  })
  
  output$plotTREE1 <- renderPrint({
    complete_test = cleaned_data()$test
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit10$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit10$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    c_test = predict(preProc, c_test)
    
    testPred_10 = predict(fit10, c_test, type = "prob")
    best_cutoff =  0.155506
    preds10 = ifelse(testPred_10$spam > best_cutoff, "spam", "ham")
    confusionMatrix(preds10, c_test$type)
  })
  
  
  output$plotNNET1 <- renderPrint({
    complete_test = cleaned_data()$test
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit12$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit12$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    c_test = predict(preProc, c_test)
    
    testPred_12 = predict(fit12, c_test, type = "prob")
    best_cutoff =  0.2099552
    preds12 = ifelse(testPred_12$spam > best_cutoff, "spam", "ham")
    confusionMatrix(preds12, c_test$type)
  })
  
  
  output$KNNroc <- renderImage({
    filename <- "roc_fit4.png"
    list(src = filename,
         alt = paste("Image number"))
    
  }, deleteFile = FALSE)
  
  output$LDAroc <- renderImage({
    filename <- "roc_fit9.png"
    list(src = filename,
         alt = paste("Image number"))
    
  }, deleteFile = FALSE)
  
  output$SVMroc <- renderImage({
    filename <- "roc_fit7.png"
    list(src = filename,
         alt = paste("Image number"))
    
  }, deleteFile = FALSE)
  
  output$TREEroc <- renderImage({
    filename <- "roc_fit10.png"
    list(src = filename,
         alt = paste("Image number"))
    
  }, deleteFile = FALSE)
  
  output$NNETroc <- renderImage({
    filename <- "roc_fit12.png"
    list(src = filename,
         alt = paste("Image number"))
    
  }, deleteFile = FALSE)

  load("knn_fit4.rda")
  load("lda_fit9.rda")
  load("svm_fit7.rda")
  load("logit_fit3.rda")
  load("tree_fit10.rda")
  load("nnet_fit12.rda")
  load("preProc.Rdata")
  
  output$fitKNN <- renderPrint({fit4})
  output$fitLDA <- renderPrint({fit9})
  output$fitSVM <- renderPrint({fit7})
  output$fitLOGIT <- renderPrint({fit3})
  output$fitTREE <- renderPrint({fit10})
  output$fitNNET <- renderPrint({fit12})
  
  output$preprocess <- renderText({paste("PreProcess: scale, center")})
  
  
  

  
  output$calcSVM <- renderPrint({
    #browser()
    complete_test = cleaned_msg()$msg
    #browser()
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit7$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit7$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    #c_test = predict(preProcess(c_test, method = c("center", "scale") ), c_test)
    c_test = predict(preProc, c_test)
    testPred_7 = predict(fit7, c_test, type = "prob")
    if (testPred_7$spam > 0.1788879){
      print("SPAM")
    }
    else{
      print("HAM")
    }
  })
  
  output$calcLDA <- renderPrint({
    complete_test = cleaned_msg()$msg
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit9$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit9$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    #c_test = predict(preProcess(c_test, method = c("center", "scale") ), c_test)
    c_test = predict(preProc, c_test)
    testPred_9 = predict(fit9, c_test, type = "prob")
    if (testPred_9$spam > 0.0001217802){
      print("SPAM")
    }
    else{
      print("HAM")
    }
  })
  
  output$calcKNN <- renderPrint({
    complete_test = cleaned_msg()$msg
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit4$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit4$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    #c_test = predict(preProcess(c_test, method = c("center", "scale") ), c_test)
    c_test = predict(preProc, c_test)
    testPred_4 = predict(fit4, c_test, type = "prob")
    if (testPred_4$spam > 0.08012821){
      print("SPAM")
    }
    else{
      print("HAM")
    }
  })
  
  output$calcTREE <- renderPrint({
    complete_test = cleaned_msg()$msg
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit10$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit10$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    #c_test = predict(preProcess(c_test, method = c("center", "scale") ), c_test)
    #names(c_test)[names(c_test) == "sorry"] = "sorry,"
    c_test = predict(preProc, c_test)
    #names(c_test)[names(c_test) == "sorry,"] = "sorry"
    testPred_10 = predict(fit10, c_test, type = "prob")
    if (testPred_10$spam > 0.155506){
      print("SPAM")
    }
    else{
      print("HAM")
    }
  })
  
  output$calcNNET <- renderPrint({
    complete_test = cleaned_msg()$msg
    #complete_test = test
    tempNames = data.frame( matrix(0L, ncol = (length(fit12$coefnames)), nrow = dim(complete_test)[1]) )
    colnames(tempNames) = c(fit12$coefnames)
    c_test = left_join(complete_test, tempNames, by = NULL, copy = FALSE)
    c_test[is.na(c_test)] = 0
    c_test = predict(preProc, c_test)
    #c_test = predict(preProcess(c_test, method = c("center", "scale") ), c_test)
    
    testPred_12 = predict(fit12, c_test, type = "prob")
    if (testPred_12$spam > 0.2099552){
      print("SPAM")
    }
    else{
      print("HAM")
    }
  })
  
  
  
  
  
  
  
  
  cleaned_msg <- eventReactive(input$Enter, {
    validate(
      need(input$onem != "", "Please enter the message")
    )
    
    #browser()
    
    #inFile = input$file1
    full = tibble(
      sms = input$onem,
      type = "ham"
    )
    
    full$type = factor(full$type) #switch from string to factor for simpler access/manipulation
    
    corp = VCorpus( VectorSource(full$sms) ) # make corpus
    
    cleaned_corp = corp #edit a copy of the corp
    #create the toSpace content transformer
    toSpace = content_transformer(function(x, pattern) {return (gsub(pattern, " ", x))})
    cleaned_corp = tm_map(cleaned_corp, toSpace, "!")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\"")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "#")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "$")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "%")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "&")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\(") # a pain in the neck
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\)")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\*")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\+")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\-")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\.")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\"")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "/")
    cleaned_corp = tm_map(cleaned_corp, toSpace, ":")
    cleaned_corp = tm_map(cleaned_corp, toSpace, ";")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "<")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "=")
    cleaned_corp = tm_map(cleaned_corp, toSpace, ">")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\?")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "@")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\[")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\]")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\^")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\_")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\{")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\|")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\}")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "~")
    cleaned_corp = tm_map(cleaned_corp, toSpace, "\\.")
    
    
    
    
    #APPLY DEFAULT TRANSFORMATIONS
    #1.
    #remove numbers
    cleaned_corp = tm_map(cleaned_corp, removeNumbers)
    
    wcaps_corp = cleaned_corp #save it just in case beofre getting rid of caps
    #3.
    #remove caps letters.
    #cleaned_corp = tm_map(cleaned_corp, tolower)
    #NOTE: ha! it looks like tolower, being a non standard transformation, transforms messages NOT into text files. dunno why. fixed by making it a content_transformer()
    cleaned_corp = tm_map(cleaned_corp, content_transformer(tolower))
    #Later we will include number of caps, number of stop words, etc as parameters.
    #4.
    #Get rid of stop words
    #cleaned_corp = tm_map(cleaned_corp, removeWords, stopwords())
    #5.
    #Get rid of tons of space bars
    cleaned_corp = tm_map(cleaned_corp, stripWhitespace)
    #Remove Punctuation
    cleaned_corp = tm_map(cleaned_corp, removePunctuation)
    #6.
    #Stem the words
    #Converting verbs to their root word. Example: "fighting" or "figths" becomes "fight".
    #NOTE: it doesn't turn fought to fight
    #Downloaded the first stemming package I found - "SnowballC". There is also "textstem", but I don't know.
    #wordStem(c("fighting", "fought", "fight", "fights", "fighter"))
    cleaned_corp = tm_map(cleaned_corp, stemDocument)
    
    
    
    
    
    #print("done")
    #})
    #output$calculateA <- renderPrint({
    dtm_text = DocumentTermMatrix(cleaned_corp)
    #findFreqTerms(dtm_text, 100, 30000) #interesting. I bet free, get, text, stop, now, reply, today are primarily from spam messages
    
    #Very high sparsity, gotta remove some of that. I am using a rather low cutoff value here - 1% - reason is that later I want to combine multiple words into one variable using PCA.
    m_text = as.matrix(dtm_text) #make it into a matrix to extract the words and to take a look
    best_words = colnames(m_text)
    
    #DTM with most frequent words only
    dtm_freq_text = dtm_text[, best_words]
    df_words = as.data.frame(as.matrix(dtm_freq_text))
    #Make a data frame
    df_words = bind_cols(full[,1] %>% as.character() %>% as.data.frame(),df_words)
    names(df_words)[names(df_words) == "."] = "sms"
    
    
    
    
    temp = full$sms %>% as.character() %>% as.data.frame()
    #names(temp) = NULL
    #1. add number of characters
    x = data.frame(apply(temp, 2, nchar))
    colnames(x) = c("nchars")
    df_words = bind_cols(x, df_words)
    #2. add number of block capitols
    #note: GOODNESS GRACIOUS, this thing didn't work for an hour
    numOfCaps = function(str){ #function that counts caps
      ldply(str_match_all(str,"[A-Z]"), length)  
    }
    x = data.frame(apply(temp, 2, numOfCaps))
    colnames(x) = c("ncaps")
    df_words = bind_cols(x, df_words)
    
    #3. add number of symbols
    numOfSymbols = function(str){ #function that counts symbols
      ldply(str_match_all(str,"[!-@]"), length)  
    }
    x = data.frame(apply(temp, 2, numOfSymbols))
    colnames(x) = c("nsymbs")
    df_words = bind_cols(x, df_words)
    #print("done")
    #})
    #output$calculateB <- renderPrint({
    
    answer = df_words
    
    list(msg = answer)
  })
  
})
