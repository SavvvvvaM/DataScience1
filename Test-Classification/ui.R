#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinycssloaders)

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("Spam/Ham SMS Classification."),
  h6("c/e: SavvaMorozov", align = "right"), 
  img(src = "my_symbol.png", height = 30, width = 30, align = "right"),
  
  h4(" "),
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      h4("Upload your CSV file to see models' performance"),
      fileInput("file1", "Choose CSV File",
                accept = c(
                  "text/csv",
                  "text/comma-separated-values,text/plain",
                  ".csv")
      ),
      h5("Does your csv have a header?"),
      checkboxInput("header", "Header", TRUE),
      tags$hr(),
      img(src = "spamham.jpg", height = 150, width = 200)
    ),
    # Show a plot of the generated distribution
    mainPanel(
      tabsetPanel(
        tabPanel("Description",
                 h3("Project Description"),
                 h4("   On this website you will find a set of classification models used to distinguish between spam/non-spam SMS messages. The SMS Spam/Ham dataset, taken from Kaggle.com, was used to train the following models: Support Vector Machine (SVM), Linear Discriminant Analysis (LDA), K Nearest Neighbors (KNN), Decision Tree, and a Neural Network. "),
                 h4("   ", span("Model Info", style = "color:blue"), " tab contains descriptions and details regarding the performance of the aforementioned models."),
                 h4("   ", span("Your Data", style = "color:blue"), "tab allows the user to upload his or her own dataset to see models’ performances on this dataset."),
                 h4("   ", span("You Enter", style = "color:blue"), " tab prompts the user to enter his or her own message example to see whether it would be classified as Spam or Ham." ),
                 h4("   ", span("Codes", style = "color:blue"), "tab lists MarkDown versions of all programming codes that were used in this project.", "The projected was written in R programming language, but in the interest of learning data science and comparing the differences between main programming languages used in the field, I’ve also completed the same project in Python and MATLAB.")
                 ),
        tabPanel( "Model Info", #textOutput("preprocess"),
          tabsetPanel(
            tabPanel("SVM", 
                     actionButton("showSVM", "What is SVM?"),
                     h4("Model Description: "), 
                     
                     
                     #verbatimTextOutput("dataInfo"),
                     
                     p("Pre-Processing methods: the data was centered (subtract the mean) and scaled (devide by SD)"),
                     withSpinner(verbatimTextOutput("fitSVM")), 
                     "Roc Curve plot. Area under curve is 0.9913",
                     withSpinner(imageOutput("SVMroc"))
                     ),
            tabPanel("LDA", 
                     actionButton("showLDA", "What is LDA?"),
                     h4("Model Description: "),
                     p("Pre-Processing methods: the data was centered (subtract the mean) and scaled (devide by SD)"),
                     withSpinner(verbatimTextOutput("fitLDA")),
                     "Roc Curve plot. Area under curve is 0.9881",
                     withSpinner(imageOutput("LDAroc"))
                     ), 
            
            tabPanel("KNN", 
                     actionButton("showKNN", "What is KNN?"),
                     h4("Model Description: "), 
                     p("Pre-Processing methods: the data was centered (subtract the mean) and scaled (devide by SD)"),
                     withSpinner(verbatimTextOutput("fitKNN")),
                     "Roc Curve plot. Area under curve is 0.9927",
                     withSpinner(imageOutput("KNNroc")) 
            ),
            tabPanel("Tree", 
                     actionButton("showTREE", "What is a Decision Tree?"),
                     h4("Model Description: "),
                     p("Pre-Processing methods: the data was centered (subtract the mean) and scaled (devide by SD)"),
                     withSpinner(verbatimTextOutput("fitTREE")),
                     "Roc Curve plot. Area under curve is 0.9603",
                     withSpinner(imageOutput("TREEroc"))
            ),
            tabPanel("NNet", 
                     actionButton("showNNET", "What is Neural Network?"),
                     h4("Model Description: "),
                     p("Pre-Processing methods: the data was centered (subtract the mean) and scaled (devide by SD)"),
                     withSpinner(verbatimTextOutput("fitNNET")),
                     "Roc Curve plot. Area under curve is 0.9998",
                     withSpinner(imageOutput("NNETroc"))
            )
            
            
            #,
            #
            #tabPanel("Logit", "It's complicated. How about we just don't talk about it, it's here just cause I did fit it, you know", 
             #        verbatimTextOutput("fitLOGIT") )
          )
        ),
        tabPanel("Your Data", 
                 h4("Please upload your dataset in the form of a .csv file."),
                 h5("1st column - true classifiers (\"spam\" or \"ham\",  with quotation marks)"),
                 h5("2nd column - sms messages (also in quotation marks)."),
                 h5("The version where the user can upload only the messages and get spam/ham classifiers as a result is to be implemented."),
                 tags$br(),
                 p("Below are confusion matrix results obtained from processing your dataset."),
                 tabsetPanel(
                   tabPanel("SVM", actionButton("showCUT", "What are cutoffs?"), br(),"Best Specificity+Sensitivity cutoff", withSpinner(verbatimTextOutput("plotSVM1")), 
                            "Cutoff that maximizes validation set’s accuracy:", withSpinner(verbatimTextOutput("plotSVM2")),
                            "This cutoff was acquired manually by determining the threshold that gave the best accuracy results on the testing dataset used to validate the model.
                            I believe I overfitted the validation set this way, since the performance of this cutoff is poor."
                            ),
                   tabPanel("LDA", actionButton("showCUT", "What are cutoffs?"), br(),"Best Specificity+Sensitivity cutoff:", withSpinner(verbatimTextOutput("plotLDA1")), 
                            "Cutoff that maximizes validation set’s accuracy:", withSpinner(verbatimTextOutput("plotLDA2")),
                            "This cutoff was acquired manually by determining the threshold that gave the best accuracy results on the testing dataset used to validate the model.
                            I believe I overfitted the validation set this way, since the performance of this cutoff is poor."
                            ),
                   tabPanel("KNN", actionButton("showCUT", "What are cutoffs?"), br(),"Best Specificity+Sensitivity cutoff:", withSpinner(verbatimTextOutput("plotKNN1")), 
                            "Cutoff that maximizes validation set’s accuracy:", withSpinner(verbatimTextOutput("plotKNN2")),
                            "This cutoff was acquired manually by determining the threshold that gave the best accuracy results on the testing dataset used to validate the model.
                            I believe I overfitted the validation set this way, since the performance of this cutoff is poor."
                            ),
                   tabPanel("Tree", actionButton("showCUT", "What are cutoffs?"), br(),"Best Specificity+Sensitivity cutoff:", withSpinner(verbatimTextOutput("plotTREE1")) 
                            ),
                   tabPanel("NNet", actionButton("showCUT", "What are cutoffs?"), br(),"Best Specificity+Sensitivity cutoff:", withSpinner(verbatimTextOutput("plotNNET1")) 
                   )
                   
                   #,
                   #tabPanel("Logit", "oooops this one doesn't work.", 
                  #          "Why? Because I was performing logit with a default package and not caret.",
                   #         "Also because Logit changes the variables, hence matching them is next to impossible")
                  )  
                 ),
        tabPanel("You Enter", 
                 h4("Enter your message below and see if the models think it is spam!"),
                 textInput("onem", "Type your message:", value = ""),
                 actionButton("Enter", "Enter"),
                 tabsetPanel(
                   tabPanel("SVM", withSpinner(verbatimTextOutput("calcSVM"))),
                   tabPanel("LDA", withSpinner(verbatimTextOutput("calcLDA"))),
                   tabPanel("KNN", withSpinner(verbatimTextOutput("calcKNN"))),
                   tabPanel("Tree", withSpinner(verbatimTextOutput("calcTREE"))),
                   tabPanel("NNet", withSpinner(verbatimTextOutput("calcNNET")))
                 )  
        ),
        tabPanel("R vs Python vs MATLAB",
                 br(),
                 p("The goal of this project was to compare the implementation of various data science methods and practices in R, Python, and MATLAB programming languages. 
                   Step-by-step comparison article is to be written."),
                 tabsetPanel(
                   tabPanel("R Code", includeMarkdown("TEST_CLASSIFICATION.md")
                   ),
                   tabPanel("Python Code", includeMarkdown("Text_Processing/Text_Processing.md")
                   ),
                   tabPanel("MATLAB Code", "Never mind the next line, the codes starts after that:", 
                            HTML(knitr::knit2html("Classification_Matlab.html", fragment.only = TRUE)),
                            HTML(readLines("Classification_Matlab.html"))
                            #includeHTML("Classification_Matlab.html")
                   )
                 )
        )
      )
    )
  )
))
