#IA Tensorflow Test

ア マン カンノツ ウンデルスタンヅ ゼ アルツ ヘ イス スツヂイング イフ ヘ オンリ ロウク フオル ゼ エンヅ 
レズルツ ヰザオウツ タキング ゼ チメ ト =デレヴエ デウペリ イント ゼ レゾニング オフ ゼ スツヂ。

ミャモト ムサシ

##Summary


This is a test programme whose goal is to create an IA which goal will be to chose a payload according to a state.
To do so, we created a fake environment composed by two states and two possible actions (action related to the two names of payload provided). 
You can find them in the hastable, at the beginning of the programme. 
To run the programme, you may launch it threw a linux terminal like any python programme, or you might launch it through a python IDE like PyCharm.


##Usefull Tools

###Tensorflow board (tensorboard)

To launch tensorboard, write the following command, in your virtual environment :

    tensorboard --logdir log_dir/

Then, in your web browser, go to :

    http://localhost:6006

You'll be able to see different graph related to the accuracy of the IA

                                                                                                      
                                                            ████████                                  
                                                          ▒▒▒▒▒▒██▒▒▒▒                                
                                                        ▓▓░░▒▒▓▓  ░░▒▒██                              
                                                      ██░░▒▒▓▓▒▒▓▓  ░░▒▒██                            
                                                    ██░░▒▒▓▓▒▒▒▒▒▒▓▓  ░░▒▒██                          
                                                  ██░░▒▒▓▓▒▒▒▒░░░░▒▒▓▓  ░░▒▒▓▓      ██████            
                                                ▓▓░░▒▒▓▓▒▒▒▒░░░░░░░░▒▒▓▓  ░░▒▒▓▓  ▓▓░░▓▓▓▓▓▓          
                                                ▓▓▒▒▓▓▒▒▒▒▒▒░░░░░░░░░░░░▓▓  ░░▒▒▓▓░░░░░░▓▓██          
                                                ▓▓▓▓▓▓▒▒░░  ▒▒░░░░░░  ░░▒▒▓▓  ░░▒▒▓▓░░▒▒▒▒██          
                                                ▓▓▓▓▓▓▓▓▒▒░░  ▒▒░░░░░░  ░░▒▒▒▒  ░░▒▒▓▓▒▒▓▓░░          
                                                  ██▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░  ░░▒▒▒▒  ░░▒▒▓▓              
                                                    ██▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░  ░░▒▒▒▒  ░░▒▒██            
                                                      ██▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░  ░░▒▒▓▓  ░░▒▒██          
                                                        ▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░▒▒▓▓  ░░▒▒██        
                                                          ▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░▒▒▓▓  ░░▒▒██      
                                                            ▓▓▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░  ░░▒▒▓▓  ░░▒▒██    
                                                              ▓▓▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░  ░░▒▒▓▓  ░░▒▒██  
                                                            ▓▓░░▓▓▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░░░░░▒▒▓▓  ▓▓▓▓██
                                                          ▓▓░░░░░░▓▓▓▓▓▓▓▓▒▒░░  ▒▒░░░░░░  ▒▒▒▒▓▓  ████
                                                        ▓▓░░░░░░▒▒▒▒▓▓▓▓▓▓▓▓▒▒░░  ▒▒░░  ▒▒▒▒▓▓▓▓▓▓░░██
                                                      ██░░░░░░▒▒▒▒██  ██▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓██░░▒▒██
                                                    ██  ░░░░▒▒▒▒██      ██▓▓▓▓▓▓▒▒░░  ▒▒▓▓▓▓██░░▒▒██  
                          ░░        ░░            ██░░░░░░▒▒▒▒██          ██▓▓▓▓▓▓▒▒▒▒▓▓▓▓██░░▒▒██    
                                  ░░  ░░  ░░    ██  ░░░░▒▒▒▒▓▓              ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░▒▒██      
                      ░░              ░░      ██  ░░░░▒▒▒▒▓▓                ░░▓▓▓▓▓▓▓▓▓▓░░▒▒██        
                    ░░                      ██  ░░░░▒▒▒▒▓▓                    ░░▓▓▓▓▓▓░░▒▒██          
                                          ██  ░░░░▒▒▒▒██                        ░░██▓▓██▓▓            
                                        ▓▓  ░░░░▒▒▒▒██                                                
                                      ██  ░░░░▒▒▒▒██                                                  
                                    ▓▓  ░░░░▒▒▒▒██                                                    
                                  ▓▓  ░░░░▒▒▒▒██                                                      
                              ░░▓▓░░░░░░▒▒▒▒██                                                        
                            ░░▓▓░░░░░░▒▒▒▒▓▓                                                          
                          ░░▒▒░░░░░░▒▒▒▒▓▓                                                            
                          ▒▒░░░░░░▒▒▒▒▓▓                                                              
                        ██░░░░░░▒▒▒▒▓▓                                                                
                      ██  ░░░░▒▒▒▒▓▓                                                                  
                    ██  ░░░░▒▒▒▒▓▓                                                                    
                  ██  ░░░░▒▒▒▒▓▓                                                                      
                ▓▓  ░░░░▒▒▒▒▓▓                                                                        
              ██  ░░░░▒▒▒▒▒▒                                                                          
            ▓▓░░░░░░▒▒▒▒██                                                                            
        ████▓▓▓▓░░▒▒▒▒██                                                                              
      ██▒▒  ▒▒▓▓▓▓▒▒██                                                                                
      ▓▓▒▒  ▒▒▒▒▓▓██                                                                                  
      ▓▓▒▒▒▒▒▒▓▓██                                                                                    
      ██▓▓▓▓▓▓▓▓██                                                                                    
      ░░▓▓▓▓▓▓▓▓░░                                                                                    



