############# Part one #####################
library(XML)
# who posted it
getName =
  function(post) {
    userInfo = getNodeSet(post, ".//div[@class = 'user-details']")
    Name = getNodeSet(userInfo[[1]], ".//a/text()")   
    if (length(Name) == 0) 
      post.user = NA
    else  
      post.user = xmlValue(Name[[1]])
    post.user
  }

getReputation = 
  function(post) {
    # the reputation level of the poster
    repNode = getNodeSet(post, ".//span[@class = 'reputation-score']")
    if (length(repNode)==0)
      post.reputation = NA
    else
      rep = xmlValue(repNode[[1]])
    # if the reputation is simplified as **K, go back to extract the title and 
    # get the exact number of reputation (if any)
    if (grepl("k", rep, ignore.case = TRUE)) {
      tmp = xmlGetAttr(repNode[[1]], "title")
      repu = unlist(strsplit(tmp, " "))
      post.reputation = repu[3]
      if (is.na(post.reputation))
        post.reputation = rep}   
    else
      post.reputation = rep
    # remove the comma
    post.reputation = gsub(",", "", post.reputation)
    post.reputation = as.integer(post.reputation) # convert the class from character into integer
    post.reputation
  }


getDate =
  function(post) {
    # when it was posted
    timeInfo = getNodeSet(post, ".//div[@class='user-action-time']")
    time = getNodeSet(timeInfo[[1]], ".//span/@title")
    if (length(time) == 0) 
      post.date = NA
    else  
      post.date = unlist(time)[1]
    post.date
  }

getTitle =
  function(post) {
    # the title of the post
    titleInfo = getNodeSet(post, ".//h3/a/text()")
    post.title = xmlValue(titleInfo[[1]])
    post.title
  }

getView =
  function(post) {
    # the current number of views for the post
    viewNode = getNodeSet(post, ".//div[@class='views ']")
    if (length(viewNode)==0)
      post.view = NA 
    else
    {
      view = xmlValue(viewNode[[1]], trim = TRUE)
      post.views = unlist(strsplit(view, " "))[1]}
    post.views = as.integer(post.views)
    post.views
  }

getAnswer =
  function(post) {
    # the current number of answers for the post
    answerNode = getNodeSet(post, ".//div[@class='status answered']/strong")
    # if no one answers, set the answer count = 0
    if (length(answerNode)==0)
      post.answers = 0
    else 
      post.answers = xmlValue(answerNode[[1]], trim = TRUE)
    post.answers = as.integer(post.answers)
    post.answers
  }

getVote =
  function(post) {
    # the vote "score" for the post
    voteNode = getNodeSet(post, ".//span[@class='vote-count-post ']/strong")
    post.votes = xmlValue(voteNode[[1]], trim = TRUE)
    post.votes = as.integer(post.votes)
    post.votes
  }

getUrl =
  function(post) {
    # the URL for the page with the post, answers and comments
    urlNode = getNodeSet(post, ".//a[@class='question-hyperlink']/@href")
    post.url = getRelativeURL(unlist(urlNode), docName(doc))
    post.url
  }

getId =
  function(post) {
    # the id (a number) uniquely identifying the post
    idNode = getNodeSet(post, "@id")
    post.id = unlist(strsplit(unlist(idNode), "-"))[3]
    post.id
  }

getTag =
  function(post) {
    # the tags of the posts
    tagList = xpathSApply(post, ".//a[@rel='tag']", xmlValue)
    post.tags = paste(tagList, collapse=";")
    post.tags
  }


# write a function to generate one row of data frame for one post
postInfo = 
  function(post) {
    # put results into one row of a data frame
    data.frame(
      id = getId(post),
      date = getDate(post),
      tags = getTag(post),
      title = getTitle(post),
      url = getUrl(post),
      views = getView(post),
      votes = getVote(post),
      answers = getAnswer(post),
      user = getName(post),
      reputation = getReputation(post),
      stringsAsFactors = FALSE
    )
  }

# write a function to get all info for one page, the output is a data frame
getPage =
  function(doc) {
    d = xpathApply(doc, "//div[@class='question-summary']", postInfo)
    cur = do.call(rbind, d)
    cur   
  }


# write a function to get the URL
getNextURL =
  function(doc) {
    # find the relative url of next page
    nextNode = getNodeSet(doc, "//a[@rel='next']/@href")
    nxt = unlist(nextNode)[1]
    # get the complete url
    nextUrl = getRelativeURL(nxt, docName(doc))
    nextUrl
  }

# the top level function
stackOverFlowInfo =
  function(tag = "r", pageNum=-1)
  {
    u = paste0("http://stackoverflow.com/questions/tagged/",tag)
    ans = NULL
    page = 1
    while(page < pageNum + 1|pageNum<0) {
      doc = htmlParse(u)
      cur = getPage(doc)
      ans = rbind(ans, cur)
      saveRDS(ans, "ans.rds")
      u = getNextURL(doc)
      if(length(u) == 0)
        break
      page = page + 1
    }   
    return(ans)
  }
# scrape 500 pages, so that we have 7500 rows 
ans = stackOverFlowInfo("r", 500)
saveRDS(ans, "ans.rds")
rownames(ans) = 1:nrow(ans)
head(ans)

############# Part two #####################


# write a function to get all post links from one overview page, the output is a list of urls
getPostLinks =
  function(doc){
    postList = getNodeSet(doc, "//a[@class='question-hyperlink']/@href")
    postUrls=rep('',length(postList))
    for(i in 1:length(postList)){
      postUrls[i] = getRelativeURL(unlist(postList)[i], docName(doc))
    }
    postUrls 
  }

#Get info from a question
getQuestion =
  function(postUrl) {
    #postUrl="http://stackoverflow.com/questions/32181962/how-to-add-two-arrays-in-one-soap-object-for-billing-and-shipping-address-in-mag"
    post=htmlParse(postUrl)
    
    postClass=getNodeSet(post, "//div[@class='question']")
    
    postCell=getNodeSet(postClass[[1]], "//td[@class='postcell']")
    
    #1 type of entry
    postCell.type='post'
    
    #2 the user
    postOwnerInfo = getNodeSet(postCell[[1]], "//td[@class='post-signature owner']")
    userInfo = getNodeSet(postOwnerInfo[[1]], ".//div[@class = 'user-details']")
    Name = getNodeSet(userInfo[[1]], ".//a/text()")   
    if (length(Name) == 0) {
      postCell.user = NA
    }else{
      postCell.user = xmlValue(Name[[1]]) # extract the poster's name
    }
    
    #3 userid
    Userid = getNodeSet(userInfo[[1]], ".//a/@href")
    postCell.userid = strsplit(unlist(Userid),'/')$href[3]
    
    #4 date
    dateInfo = getNodeSet(postOwnerInfo[[1]], ".//span[@class='relativetime']/@title")
    postCell.date=strsplit(unlist(dateInfo)[[1]],' ')[[1]][1]
    
    #5 user's reputation
    reputationInfo = getNodeSet(postOwnerInfo[[1]], ".//span[@class='reputation-score']")
    postCell.reputation=xmlValue(reputationInfo[[1]])
    
    #6 votes
    voteInfo=getNodeSet(postClass[[1]], ".//span[@class='vote-count-post ']")
    postCell.vote=xmlValue(voteInfo[[1]])
    
    #7 HTML content
    contentInfo=getNodeSet(postCell[[1]], ".//div[@class='post-text']")
    postCell.content=xmlValue(contentInfo[[1]])
    
    #8 Parent entry
    postCell.parent=NA
    
    #9 unique identifier for the overall post
    postidInfo=getNodeSet(postClass[[1]], "//div[@class='question']/@data-questionid")
    postCell.id=unlist(postidInfo)[1]
    
    # put results into one row of a data frame
    post_df=data.frame(
      type=postCell.type,
      user=postCell.user,
      userid=postCell.userid,
      date=postCell.date,
      reputation=postCell.reputation,
      vote=postCell.vote,
      content=postCell.content,
      parent=postCell.parent,
      id=postCell.id
    )
    
    #Comments
    parentID=unlist(getNodeSet(postClass[[1]], "//div[@class='question']/@data-questionid"))[[1]]
    comment_df=data.frame(do.call(rbind,lapply(postClass,getComment)))
    res_df=rbind(post_df,comment_df)
    return(res_df)
    
  }


#Get info from an answer
getAnswer =
  function(postUrl) {
    #postUrl="http://stackoverflow.com/questions/32181962/how-to-add-two-arrays-in-one-soap-object-for-billing-and-shipping-address-in-mag"
    post=htmlParse(postUrl) 
    answerd=xpathApply(post, "//div[@itemtype='http://schema.org/Answer']", getAnswerInfo)
    answerDF=do.call(rbind, answerd)
    answerDF
  }

#Get Info from a post Class 
getAnswerInfo =
  function(postCell) {
    #postCell=postClass[1]
    #1 type of entry
    postCell.type='answer'
    
    #2 the user
    userInfo = getNodeSet(postCell, ".//td[@class ='post-signature']")
    if(length(userInfo)==0){
      postCell.user = NA
    }else{
      Name = getNodeSet(userInfo[[1]], ".//a/text()")   
      if (length(Name) == 0) {
        postCell.user = NA
      }else{
        postCell.user = xmlValue(Name[[1]]) # extract the poster's name
      }
    }
    
    #3 userid
    if(length(userInfo)>0){
      userDetail=getNodeSet(userInfo[[1]], ".//div[@class = 'user-details']")
      Userid = getNodeSet(userDetail[[1]], ".//a/@href")
      if(length(Userid)>0){
        postCell.userid = strsplit(unlist(Userid),'/')$href[3]
      }else{
        postCell.userid = NA
      }
    }else{
      postCell.userid = NA
    }
    
    
    #4 date
    if(length(userInfo)>0){
      dateInfo = getNodeSet(userInfo[[1]], ".//span[@class='relativetime']/@title")
      postCell.date=strsplit(unlist(dateInfo)[[1]],' ')[[1]][1]
    }else{
      postCell.date=NA
    }
    #5 user's reputation
    if(length(userInfo)>0){
      reputationInfo = getNodeSet(userInfo[[1]], ".//span[@class='reputation-score']")
      if(length(reputationInfo)>0){
        postCell.reputation=xmlValue(reputationInfo[[1]])
      }else{
        postCell.reputation=NA
      }
    }else{
      postCell.reputation=NA
    }
    
    #6 votes
    voteInfo=getNodeSet(postCell, ".//span[@class='vote-count-post ']")
    postCell.vote=xmlValue(voteInfo[[1]])
    
    #7 HTML content
    contentInfo=getNodeSet(postCell, ".//div[@class='post-text']")
    postCell.content=xmlValue(contentInfo[[1]])
    
    #8 Parent entry for answer: Question ID
    postidInfo=getNodeSet(postCell, "//div[@class='question']/@data-questionid")
    postCell.parent=unlist(postidInfo)[1]
    
    #9 unique identifier for the overall post
    postidInfo=getNodeSet(postCell, "//div[@class='question']/@data-questionid")
    postCell.id=unlist(postidInfo)[1]
    
    # put results into one row of a data frame
    post_df=data.frame(
      type=postCell.type,
      user=postCell.user,
      userid=postCell.userid,
      date=postCell.date,
      reputation=postCell.reputation,
      vote=postCell.vote,
      content=postCell.content,
      parent=postCell.parent,
      id=postCell.id
    )
    
    #Comments
    parentID=unlist(getNodeSet(postCell, "@data-answerid"))[[1]]
    #comment_df=data.frame(do.call(rbind,getComment(postCell)))
    comment_df=getComment(postCell)
    res_df=rbind(post_df,comment_df)
    return(res_df)
  }


#Get info from a comment
getComment =
  function(postModule) {
    #postUrl="http://stackoverflow.com/questions/32181962/how-to-add-two-arrays-in-one-soap-object-for-billing-and-shipping-address-in-mag"
    #post=htmlParse(postUrl)
    commentd=xpathApply(postModule, ".//tr[@class='comment ']", getCommentInfo)
    commentDF=do.call(rbind, commentd)
    commentd=xpathApply(postModule, ".//tr[@class='comment']", getCommentInfo)
    commentDF2=do.call(rbind, commentd)
    rbind(commentDF,commentDF2)
  }

#Get Info from a post Class 
getCommentInfo =
  function(postCell) {
    #postCell=postClass[1]
    #1 type of entry
    postCell.type='comment'
    
    #2 the user
    postCell.user=NA
    userInfo = getNodeSet(postCell, ".//a[@class ='comment-user']")
    if(length(userInfo)>0){
      postCell.user=xmlValue(userInfo[[1]])
    }else{
      userInfo = getNodeSet(postCell, ".//a[@class ='comment-user owner']")
      if(length(userInfo)>0){
        postCell.user=xmlValue(userInfo[[1]])
      }
    }   
    #3 userid
    if(length(userInfo)>0){
      Userid = getNodeSet(userInfo[[1]], "..//a/@href")
      if(length(Userid)>0){
        postCell.userid = strsplit(unlist(Userid),'/')$href[3]
      }else{
        postCell.userid = NA
      }
    }else{
      postCell.userid = NA
    }    
    #4 date
    dateInfo = getNodeSet(postCell, ".//span[@class='relativetime-clean']/@title")
    if(length(dateInfo)>0){ 
      postCell.date=strsplit(unlist(dateInfo)[[1]],' ')[[1]][1]
    }else{
      postCell.date=NA
    }   
    #5 user's reputation
    if(length(userInfo)>0){
      reputationInfo = getNodeSet(userInfo[[1]], "..//a/@title")
      if(length(reputationInfo)>0){
        postCell.reputation=strsplit(unlist(reputationInfo)[[1]],' ')[[1]][1]
      }else{
        postCell.reputation=NA
      }
    }else{
      postCell.reputation=NA
    } 
    #6 votes
    postCell.vote=NA    
    #7 HTML content
    contentInfo=getNodeSet(postCell, ".//span[@class='comment-copy']")
    postCell.content=xmlValue(contentInfo[[1]])    
    #8 Parent entry for comment: Question / Answer ID
    postCell.parent=parentID    
    #9 unique identifier for the overall post
    postidInfo=getNodeSet(postCell, "//div[@class='question']/@data-questionid")
    postCell.id=unlist(postidInfo)[1]    
    # put results into one row of a data frame
    post_df=data.frame(
      type=postCell.type,
      user=postCell.user,
      userid=postCell.userid,
      date=postCell.date,
      reputation=postCell.reputation,
      vote=postCell.vote,
      content=postCell.content,
      parent=postCell.parent,
      id=postCell.id
    )
    post_df
  }


# write a function to get the URL
getNextURL =
  function(doc) {
    # find the relative url of next page
    nextNode = getNodeSet(doc, "//a[@rel='next']/@href")
    nxt = unlist(nextNode)[1]
    # get the complete url
    nextUrl = getRelativeURL(nxt, docName(doc))
    nextUrl
  }

# the top level function 
postPageScrape =
  function(tag = "r", pageNum)
  {
    u = paste0("http://stackoverflow.com/questions/tagged/",tag)
    ans = NULL
    page = 1
    while(page < pageNum + 1) {
      doc = htmlParse(u)
      #Get Urls for the post list
      postList = getPostLinks(doc)
      posti=1
      while(posti <= length(postList)){
        #Post Question
        cur = getQuestion(postList[posti])
        ans = rbind(ans, cur)
        #Answer
        cur = getAnswer(postList[posti])
        if(length(cur)>1){
          ans = rbind(ans, cur)
        }
        posti=posti+1
      }
      #Moving to Next Page of Post List
      u = getNextURL(doc)
      if(length(u) == 0)
        break
      page = page + 1
    }   
    return(ans)
  }

#Test
post_df=postPageScrape(pageNum=2)

############# Part three #####################


# read data into R
load("~/Desktop/201509-12/STA141statisticalComputing/homework/hw6/rQAs.rda")
class(rQAs)
dim(rQAs)
head(rQAs)
# 1.
# What is the distribution of the number of questions each person answered?
post.answer = subset(rQAs, type == "answer") # there are 10004 answers in total
# group by use id, get the frequency of answers for each user
tt = data.frame(table(post.answer$userid))
names(tt) = c("userid", "Freq")
# group by the frequency of answers, then we get how many users answers a certain number of questions
q = data.frame(table(tt$Freq))
names(q) = c("answerFreq", "userFreq")
library(ggplot2)
ggplot(q, aes(x = as.integer(answerFreq), y = userFreq))+
  geom_point(col = "blue") +
  xlab("Frequency of questions answered by a unique user") +
  ylab("Number of users") +
  scale_x_continuous(breaks=seq(0,80,5)) +
  scale_y_continuous(breaks=seq(0,1300,50)) +
  ggtitle("Distribution of the number of questions each person answered")

# 2.
# What are the most common tags?
# the following gives us a list of 7500 character vector which includs tags for each post
all.tags = sapply(ans$tags, function(x) strsplit(x, ";")) 
# unlist all these tags 
all.tags.unlist = unlist(all.tags)
# convert the tags into lower cases
tags.lower = tolower(all.tags.unlist)
# count each tag's frequency
tags.freq = data.frame(table(tags.lower))
names(tags.freq) = c("tag", "frequency")
# order the tag frequency in decreasing order
tags.freq = tags.freq[order(tags.freq$frequency, decreasing = TRUE),]
# subset the most common 10 tags
commonTag = tags.freq[1:10,]
commonTag

# 3.
# How many questions are about ggplot?
# in my own data frame:
numGGplot = subset(tags.freq, tag=="ggplot2"|tag=="ggplot")$frequency 
numGGplot # there are 619 posts about ggplot in my data frame

# in Duncan's data frame:
# subset all the questions
post.question = subset(rQAs, type == "question")
# in each question's text, see whether "ggplot" is mentioned
haveGGplot = sapply(post.question$text, function(x) {grepl("ggplot", x, ignore.case = TRUE)})
numGGplot2 = sum(haveGGplot)
numGGplot2

# 4.
# How many questions involve XML, HTML or Web Scraping?
# Web scraping is also known as web harvesting or web data extraction
# web-crawler, web-craping, web-mining
# in my own data frame:
webString = c("xml", "html", "web-crawler", "web-craping", "web-mining")
aboutWeb =
  function(x){
    # x is a vector of character
    # see whether x contains xml, html, web-crawler, web-craping, web-mining
    # if so, return true; otherwise return false
    tf = sapply(webString, function(string) grepl(string, x))
    any(tf)
  }

numWeb = sum(sapply(all.tags, aboutWeb))
numWeb
# in Duncan's data frame:
numWeb2 = sum(sapply(post.question$text, aboutWeb))
numWeb2

# 5.
# What are the names of the R functions referenced in the titles of the posts?
# The title information are included in the rownames (urls) of rQAs
# get all the urls from rownames
urls = rownames(post.question)
# write a function for each url, get the names of the R functions refered
referenceFunction = 
  function(url){
    # extract the words after the last slash
    pattern = ".*\\/(.*)"
    tmp = gsub(pattern, "\\1", url, ignore.case = TRUE)
    # get the title seperated by "-"
    title = strsplit(tmp, "\\.")[[1]][1]
    # get a vector of words in the title
    words = unique(unlist(strsplit(title, "-")))
    tmpList = sapply(words, function(x) {tryCatch({
      is.function(get(x))}, error=function(e){})})
    isFunction = unlist(tmpList)
    names = names(isFunction[isFunction])
    names
  }

fs = unlist(sapply(urls, referenceFunction))
funIntitle = data.frame(table(fs))
funIntitle[order(funIntitle$Freq, decreasing=TRUE),][1:10,]

function.name = unique(fs) 
length(function.name)
function.name


# 6.
# What are the names of the R functions referenced in the accepted answers and comments of the posts? 
# We can do better than we did in processing the title of the posts as there is HTML markup 
# and we can find content in code blocks.

# in the accepted answers
questionId = unique(post.answer$qid) # there are 6833 unique questions involved 
vv = subset(post.answer, qid == questionId[1])
accepted = vv[order(vv$score, decreasing=TRUE),][1]

acc = sapply(questionId, function(x) 
{
  answer = subset(post.answer, qid == x)
  answer[order(answer$score, decreasing=TRUE),]$text[1]
})

# write a function that for each text (html code chunk), find the R function mentioned in this piece of text
findFunction =
  function(x) {
    qq=unlist(strsplit(x," |\n|;"))
    pattern = ".*?([.A-Za-z1-9]*)\\(.*"
    tmp = sapply(qq, function(x){xx = gsub(pattern, "\\1", x)
                                 if (!identical(xx,x))
                                   return(xx)})
    unique(unlist(tmp))
  }

funcs = sapply(acc, findFunction)                       
funInAnswer = unlist(funcs)
funInAnswer = funInAnswer[nchar(funInAnswer)>0]
df.fun = data.frame(table(funInAnswer))
df.fun[order(df.fun$Freq, decreasing=TRUE),][1:10,]

post.comment = subset(rQAs, type == "comment")
funcs2 = sapply(post.comment$text, findFunction)
funInComment = unlist(funcs2)
funInComment = funInComment[nchar(funInComment)>0]
df.fun2 = data.frame(table(funInComment))
df.fun2[order(df.fun2$Freq, decreasing=TRUE),][1:10,]



