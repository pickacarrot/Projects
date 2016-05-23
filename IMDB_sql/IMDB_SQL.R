library(RSQLite)
im = dbConnect(SQLite(), pt) # connect with the database
dbListTables(im)
options(width = 80)

# 1.
# How many actors are there in the database? How many movies?
# how many actors
numActor = dbGetQuery(im, "SELECT COUNT(DISTINCT person_id) AS numActor 
                           FROM cast_info, role_type 
                           WHERE cast_info.role_id = role_type.id 
                           AND (role_type.role = 'actor' OR role_type.role = 'actress');") # 3492018 actors
numActor

# how many movies
numMovie = dbGetQuery(im, "SELECT COUNT(*) AS numMovie
                           FROM title, kind_type
                           WHERE title.kind_id = kind_type.id
                           AND kind_type.kind = 'movie';") # 878800 movies

numMovie

# 2. 
# What time period does the database cover?
timeSpan = dbGetQuery(im, "SELECT MIN(production_year) AS minYear, MAX(production_year) AS maxYear 
                           FROM title;") 
timeSpan

# 3.
# What proportion of the actors are female? male?
genderTable = dbGetQuery(im, "SELECT role, 
                                     100.*COUNT(DISTINCT(person_id))/ (SELECT COUNT(DISTINCT(person_id)) 
                                                                       FROM cast_info, role_type 
                                                                       WHERE cast_info.role_id = role_type.id
                                                                       AND role_type.role IN ('actor', 'actress')) AS percentage 
                         FROM cast_info, role_type
                         WHERE cast_info.role_id = role_type.id 
                         AND role_type.role IN ('actor', 'actress')
                         GROUP BY role")

genderTable

# 4.
# What proportion of the entries in the movies table are actual movies 
# and what proportion are television series, etc.?
movie_tv = dbGetQuery(im, "SELECT kind, 100.*COUNT(*)/(SELECT COUNT(*) FROM title) AS percent
                           FROM title, kind_type
                           WHERE title.kind_id = kind_type.id
                           AND kind_type.kind IN ('movie', 'tv series')
                           GROUP BY kind")

movie_tv

# 5.
# How many genres are there? What are their names/descriptions?
genres = dbGetQuery(im, "SELECT DISTINCT(movie_info.info) AS genres
                         FROM movie_info, info_type
                         WHERE movie_info.info_type_id = info_type.id
                         AND info_type.info = 'genres';")
genres

# it is obvious that there are 32 genres, we can also compute with SQL
numGenres = dbGetQuery(im, "SELECT COUNT(DISTINCT(movie_info.info)) AS numGenres
                            FROM movie_info, info_type
                            WHERE movie_info.info_type_id = info_type.id
                            AND info_type.info = 'genres';")
numGenres

# 6.
# List the 10 most common genres of movies, showing the number of movies in each of these genres.
commonGenres = dbGetQuery(im, "SELECT movie_info.info AS genres, COUNT(DISTINCT(title.id)) AS frequency
                               FROM title, movie_info, kind_type, info_type
                               WHERE title.id = movie_info.movie_id
                               AND movie_info.info_type_id = info_type.id
                               AND title.kind_id = kind_type.id
                               AND kind_type.kind = 'movie'
                               AND info_type.info = 'genres'
                               GROUP BY movie_info.info
                               ORDER BY COUNT(DISTINCT(title.id)) DESC
                               LIMIT 10;")
commonGenres

# 7.
# Find all movies with the keyword 'space'. How many are there? What are the years these were released? 
# and who were the top 5 actors in each of these movies? (nr_order)

# there are 401 movies with the keyword 'space'
numSpaceMovie = dbGetQuery(im, "SELECT COUNT(DISTINCT(title.id)) AS numSpace
                                FROM title, movie_keyword, keyword, kind_type
                                WHERE title.id = movie_keyword.movie_id
                                AND movie_keyword.keyword_id = keyword.id
                                AND title.kind_id = kind_type.id
                                AND kind_type.kind = 'movie'
                                AND keyword.keyword = 'space';") # 401 movies

# we list 20 movies with the keyword 'space' here
spaceMovie = dbGetQuery(im, "SELECT title.title, title.production_year
                             FROM title, movie_keyword, keyword, kind_type
                             WHERE title.id = movie_keyword.movie_id
                             AND movie_keyword.keyword_id = keyword.id
                             AND title.kind_id = kind_type.id
                             AND kind_type.kind = 'movie'
                             AND keyword.keyword = 'space'
                             LIMIT 20;")
spaceMovie
# the years of these space movie released range from 1911 to 2018. 
years_spaceMovie = dbGetQuery(im, "SELECT DISTINCT(title.production_year) AS distinctYear
                                   FROM title, movie_keyword, keyword, kind_type
                                   WHERE title.id = movie_keyword.movie_id
                                   AND movie_keyword.keyword_id = keyword.id
                                   AND title.kind_id = kind_type.id
                                   AND kind_type.kind = 'movie'
                                   AND keyword.keyword = 'space'
                                   ORDER BY production_year;") # 72 years from 1911 to 2018

t(years_spaceMovie) # transpose the data frame in order to display the resulst in a compact way

# the top 5 actors in each of these movies?
# in terms of billing position
top5Billing = dbGetQuery(im, "SELECT DISTINCT(name), title, title.id, nr_order
                         FROM title, movie_keyword, keyword, kind_type, cast_info, role_type, name
                         WHERE title.id = movie_keyword.movie_id
                         AND movie_keyword.keyword_id = keyword.id
                         AND title.kind_id = kind_type.id
                         AND cast_info.movie_id = title.id
                         AND cast_info.role_id = role_type.id
                         AND cast_info.person_id = name.id
                         AND kind_type.kind = 'movie'
                         AND keyword.keyword = 'space'
                         AND (role_type.role = 'actor' OR role_type.role = 'actress')
                         AND cast_info.nr_order IN (1,2,3,4,5)
                         ;")

# show part the top5Billing
top5Billing[1:25,]

# the top 5 actors in each of these movies?
# in terms of movie frequency
top5Common = dbGetQuery(im, "SELECT name.name, COUNT(title.id) AS frequency
                             FROM title, movie_keyword, keyword, kind_type, cast_info, role_type, name
                             WHERE title.id = movie_keyword.movie_id
                             AND movie_keyword.keyword_id = keyword.id
                             AND title.kind_id = kind_type.id
                             AND cast_info.movie_id = title.id
                             AND cast_info.role_id = role_type.id
                             AND cast_info.person_id = name.id
                             AND kind_type.kind = 'movie'
                             AND keyword.keyword = 'space'
                             AND (role_type.role = 'actor' OR role_type.role = 'actress')
                             GROUP BY cast_info.person_id
                             ORDER BY COUNT(title.id) DESC
                             LIMIT 5;")

top5Common

# 8.
# Has the number of movies in each genre changed over time? 
# Plot the overall number of movies in each year over time, and for each genre.
movieInEachYear = dbGetQuery(im, "SELECT title.production_year AS year, COUNT(*) AS frequency
                             FROM title, kind_type
                             WHERE title.kind_id = kind_type.id
                             AND kind_type.kind = 'movie'
                             AND title.production_year IS NOT NULL
                             GROUP BY title.production_year;") # 141 years 
movieInEachYear[1:10,]

# plot the number of movies in each year
library(ggplot2)
ggplot(data = movieInEachYear, aes(x = year, y = frequency)) +
  geom_point(color = "blue") +
  geom_line(color = "red") +
  scale_x_continuous(breaks=seq(1870, 2025, 5)) +
  theme(axis.text.x = element_text(angle=90, size = 10)) +
  ggtitle("Overall number of movies across the years")

# each year, the number of movies in each genre
movieInEachGenre = dbGetQuery(im, "SELECT title.production_year AS year, 
                                          movie_info.info AS genres, COUNT(*) AS frequency
                              FROM title, kind_type, movie_info, info_type
                              WHERE title.kind_id = kind_type.id
                              AND kind_type.kind = 'movie'
                              AND movie_info.info_type_id = info_type.id
                              AND movie_info.movie_id = title.id
                              AND title.production_year IS NOT NULL
                              AND info_type.info = 'genres'
                              GROUP BY title.production_year, movie_info.info;") # 3022

# plot 
ggplot(data = movieInEachGenre, aes(x = year, y = frequency)) +
  geom_point(aes(color = genres), size = 1) +
  scale_x_continuous(breaks=seq(1875, 2025, 20)) +
  theme(legend.position="none", axis.text.y = element_text(size=5), strip.text = element_text(size=6.5),
        plot.title = element_text(size = rel(1)), axis.text.x = element_text(angle=90, size=5.5)) +
  facet_wrap(~ genres, nrow = 4) +
  ggtitle("Overall number of movies across the years for each genre")

# 9.
# Who are the actors that have been in the most movies? List the top 20.
# through SQL
actorsWithMostMovie = dbGetQuery(im, "SELECT cast_info.person_id, name.name, COUNT(DISTINCT(title.id)) AS frequency
                                      FROM title, cast_info, role_type, name, kind_type
                                      WHERE title.id = cast_info.movie_id
                                      AND title.kind_id = kind_type.id
                                      AND cast_info.role_id = role_type.id
                                      AND name.id = cast_info.person_id
                                      AND kind_type.kind = 'movie'
                                      AND (role_type.role = 'actor' OR role_type.role = 'actress')
                                      GROUP BY cast_info.person_id
                                      ORDER BY COUNT(DISTINCT(title.id)) DESC
                                      LIMIT 20;")
actorsWithMostMovie

# through R 
# read the following tables into R
title = dbReadTable(im, "title")
justMovie = subset(title, kind_id == 1) # subset "title" into a table with movie only
cast_info = dbReadTable(im, "cast_info")
justActor = subset(cast_info, role_id == 1|role_id == 2) # subset "cast_info" into a table with actor and actress only
justMovieActor = subset(justActor, movie_id %in% justMovie$id) # some actors appear more than once in a movie
name = dbReadTable(im, "name")

# some actors appear in one movie more than once
# remove the duplicate movie within one actor's movies
library(dplyr)
justMovieActor2 = justMovieActor %>%
                  group_by(person_id) %>%
                  filter (! duplicated(movie_id))

# group by person_id and order by movies frequency  
tt = data.frame(table(justMovieActor2$person_id))
top20 = tt[order(tt$Freq, decreasing = TRUE),][1:20,]
# find actors' names according to their person id
top20.names = sapply(top20[,1], function(x) {subset(name, id == x)[,"name"]})
actorsWithMostMovie.r = data.frame(personID = as.integera(as.character((top20[,1]))), name = top20.names, frequency = top20[,"Freq"])
actorsWithMostMovie.r

# 10.
# Who are the actors that have had the most number of movies with "top billing", i.e., 
# billed as 1, 2 or 3? For each actor, also show the years these movies spanned?
# through SQL
actors_most_topbilling = dbGetQuery(im, "SELECT name, COUNT(DISTINCT(title.id)) AS Frequency, 
                                                      MIN(production_year) AS minYear, MAX(production_year) AS maxYear
                                         FROM title, kind_type, cast_info, role_type, name
                                         WHERE title.kind_id = kind_type.id       
                                         AND cast_info.movie_id = title.id
                                         AND cast_info.role_id = role_type.id
                                         AND cast_info.person_id = name.id
                                         AND kind_type.kind = 'movie'
                                         AND (role_type.role = 'actor' OR role_type.role = 'actress')
                                         AND cast_info.nr_order IN (1,2,3)
                                         GROUP BY cast_info.person_id
                                         ORDER BY COUNT(DISTINCT(title.id)) DESC
                                         LIMIT 5;")

actors_most_topbilling

# through R
# subset justMovieActor2 with top bills, i.e. nr_order = 1,2,3
justMovieActor2_topBill = subset(justMovieActor2, nr_order %in% c(1,2,3)) 
# group by person id and find the most frequent 5 id
tmp = data.frame(table(justMovieActor2_topBill$person_id))
top5_bill = tmp[order(tmp$Freq, decreasing = TRUE),][1:5,]
# write a function to find the name, minimum and maximum movie year for an actor given his/her id
findYearSpanName = function(actorId) {
  # input: an actor's id
  # output: this actor's name and his/her movies' time span
  movies = subset(justMovieActor2_topBill, person_id == actorId)
  years = subset(justMovie, id %in% movies$movie_id)$production_year
  actor_name = subset(name, id == actorId)[,'name']
  c(actor_name, min(years), max(years))
}
# loop findYearSpanName function over the 5 actors with most top billing movies
years_name = sapply(top5_bill[,1], findYearSpanName)
# tidy and output the final data frame
years_name2 = data.frame(t(years_name))
actors_most_topbilling.r = data.frame(cbind(years_name2,top5_bill[,'Freq']))
names(actors_most_topbilling.r) = c("name", "minYear", "maxYear", "frequency")
actors_most_topbilling.r

# 11.
# Who are the 10 actors that performed in the most movies within any given year? 
# What are their names, the year they starred in these movies and the names of the movies?
# through SQL
# create a name_year_movie table, 6938753 rows, castID is the primary key
dbGetQuery(im, "CREATE TEMPORARY TABLE name_year_movie AS
                SELECT cast_info.person_id AS personID, name.name AS name,
                       title.title AS title, title.id AS movieID, title.production_year AS year, 
                       role_type.role AS role, kind_type.kind AS kind, cast_info.id AS castID
                FROM title, cast_info, kind_type, role_type, name
                WHERE title.kind_id = kind_type.id AND kind_type.kind = 'movie'
                AND cast_info.role_id = role_type.id AND role_type.role IN ('actor', 'actress')
                AND cast_info.movie_id = title.id
                AND name.id = cast_info.person_id
                GROUP BY cast_info.person_id, cast_info.movie_id;") 


# get the top 10 actors with the most movies for a given year
top10 = dbGetQuery(im, "SELECT name, year, personID, role, kind, COUNT(movieID) AS frequency
                        FROM name_year_movie
                        GROUP BY year, personID
                        ORDER BY COUNT(movieID) DESC
                        LIMIT 10")
top10

# join the actors' name, year and the movies within this year
top10WithinYear = dbGetQuery(im, "SELECT A.name, A.year, A.personID, A.role, A.kind, name_year_movie.movieID, name_year_movie.title
                                  FROM name_year_movie, (SELECT name, year, personID, role, kind, 
                                                                COUNT(movieID) AS frequency
                                                         FROM name_year_movie
                                                         GROUP BY year, personID
                                                         ORDER BY COUNT(movieID) DESC
                                                         LIMIT 10) AS A
                                  WHERE name_year_movie.year = A.year
                                  AND name_year_movie.name = A.name;")
top10WithinYear[1:20,]              

# through r
# combine the information of person, year, movie
justMovie.sub = justMovie[,c("id", "production_year")]
names(justMovie.sub) = c("movie_id", "production_year")
justMovieActor2.sub = justMovieActor2[,c("person_id", "movie_id")]
com = merge(justMovieActor2[,c("person_id", "movie_id")], justMovie.sub, by = "movie_id")
# group by production year and person id, summarize the distinct number of movie id within each group
order_com = com%>%
            group_by(production_year, person_id) %>%
            summarize(freq = n_distinct(movie_id))
# select the 10 year and person id combinations which have the most movies
top10.r = order_com[order(order_com$freq, decreasing = TRUE),][1:10,]
# find the corresponding actors' names
top10.names = sapply(top10.r$person_id, function(x) {subset(name, id == x)[,"name"]})
top10.r = cbind(top10.names, top10.r)
names(top10.r) = c("name", "year", "personID", "frequency")
top10.r

# write a function to find all the movies given an actor' id and a given year
findMovies = function(actorId.r, year.r) {
  movie_id = subset(justMovieActor2, person_id == actorId.r)$movie_id
  top10.r.actor = top10.r[top10.r$personID == actorId.r,]
  movieInTheYear = subset(subset(justMovie, id %in% movie_id),  production_year == year.r)[,"title"]
  len = length(movieInTheYear)
  data.frame(name = rep(unique(top10.r.actor$name), len), year = rep(year.r, len), 
             personID = rep(actorId.r, len), title = movieInTheYear)
}
# loop the function findMovies over the 10 year and id combinations
top10movie.r = lapply(data.frame(t(top10.r[,c(2,3)])), function(x) findMovies(x[2], x[1]))
# tidy and output the results
top10WithinYear.r = do.call(rbind, top10movie.r)
rownames(top10WithinYear) = NULL
top10WithinYear.r[1:20,]


# 12.
# Who are the 10 actors that have the most aliases (i.e., see the aka_names table).
# through SQL
actor_with_most_aliase = dbGetQuery(im, "SELECT name.name, cast_info.person_id AS personID, COUNT(DISTINCT(aka_name.id)) AS frequency
                                         FROM cast_info, role_type, name, aka_name
                                         WHERE cast_info.person_id = name.id
                                         AND cast_info.role_id = role_type.id
                                         AND name.id = aka_name.person_id
                                         AND role_type.role IN ('actor', 'actress')
                                         GROUP BY aka_name.person_id
                                         ORDER BY COUNT(DISTINCT(aka_name.id)) DESC
                                         LIMIT 10")
actor_with_most_aliase

# through r
# read table aka_name into R
aka_name = dbReadTable(im, "aka_name")
# only actors considered
aka_name_actors = subset(aka_name, person_id %in% justActor$person_id)
# group by person id and count frequency
aliases = data.frame(table(aka_name_actors$person_id))
# order by frequency and subset the top 10
top10aka = aliases[order(aliases$Freq, decreasing = TRUE),][1:10,]
# find the corresponding names
top10aka.names = sapply(top10aka[,1], function(x) {subset(name, id == x)[,"name"]})
# tidy the results
actor_with_most_aliase.r = cbind(top10aka.names, top10aka)
names(actor_with_most_aliase.r) = c("name", "personID","frequency")      
actor_with_most_aliase.r

# 13. Networks: Pick a (lead) actor who has been in at least 20 movies. 
# Find all of the other actors that have appeared in a movie with that person. 
# For each of these, find all the people they have appeared in a move with it. 
# Use this to create a network/graph of who has appeared with who. 
# Use the igraph or statnet packages to display this network. 
# If you want, you can do this with individual SQL commands and the process the results in R to generate new SQL queries. 
# In other words, don't spend too much time trying to create clever SQL queries if there is a more direct way to do this in R.

# pull out all the actors who have been in at least 20 movies
actorIn20Movies = dbGetQuery(im, "SELECT cast_info.person_id, name.name, COUNT(DISTINCT(title.id)) AS frequency
                             FROM title, cast_info, role_type, kind_type, name
                             WHERE title.kind_id = kind_type.id
                             AND cast_info.person_id = name.id
                             AND title.id = cast_info.movie_id
                             AND cast_info.role_id = role_type.id
                             AND kind_type.kind = 'movie'
                             AND role_type.role IN ('actor', 'actress')
                             GROUP BY cast_info.person_id
                             HAVING COUNT(DISTINCT(title.id)) > 19")

# choose Boxer, John (227569) to start his relationship graph
Boxer_name = "Boxer, John"
Boxer_id = 227569

#1. pull out all movies of certain actor
find_movies = function(db, actor_id)
  # Get movies for a specific actor ID.
{
  qr = sprintf("SELECT DISTINCT(title.id) AS movieID, title.title
               FROM title, cast_info, kind_type
               WHERE title.id = cast_info.movie_id
               AND title.kind_id = kind_type.id
               AND kind_type.kind = 'movie'
               AND cast_info.person_id = %d", actor_id)
  dbGetQuery(db, qr)
}


# 2. Pull high-billed cast for all that actor's movies, we put restrictions here: nr_order in (1,2,3)
find_actors = function(db, movie_id)
  # Get cast for a specific movie ID.
{
  qr = sprintf("SELECT DISTINCT(person_id) AS actorID, name.name AS name
               FROM cast_info, name, role_type
               WHERE cast_info.person_id = name.id
               AND cast_info.role_id = role_type.id
               AND role_type.role IN ('actor', 'actress')
               AND cast_info.movie_id = %d
               AND cast_info.nr_order IN (1,2,3)", movie_id)
  dbGetQuery(db, qr)
}

# write a function for the above steps, with input of a actor_id, output of its relationship with fellows
direct_relationship = function(db, actor_id, name){
  movies = find_movies(db, actor_id)
  all_fellows = lapply(movies$movieID, function(id) find_actors(db, id))
  all_fellows = do.call(rbind, all_fellows)
  all_fellows = all_fellows[which(all_fellows[,1]!=actor_id),]
  len = nrow(all_fellows)
  relations = cbind(x = rep(actor_id, len), y = rep(name, len), all_fellows)
  relations  
}
BoxerAndfellows = direct_relationship(im, 227569, "Boxer, John")
# got a list of "relationship" between Boxer's fellows and Boxer's fellows' fellows
Boxer_fellows_fellows = apply(BoxerAndfellows[-1,], 1, 
                              function(x) {
                                id = as.integer(x[3])
                                name = as.character(x[4])
                                direct_relationship(im, id, name)})


# combine the relationship of Boxer with his fellows and his fellows with fellows' fellow
tmp = do.call(rbind, Boxer_fellows_fellows)
all_relationship = rbind(tmp, BoxerAndfellows)
# we can use name as vertices for the graph since no sharing name here
length(unique(all_relationship[,3])) == length(unique(all_relationship[,4])) # number of unique person id VS number of unique name
all_relationship = all_relationship[,c(2,4)]
# remove the duplicate relationships
ordered_relationship = data.frame(t(apply(all_relationship, 1, sort)))
# unique_relationship is the relationship data frame and will be shown as edges in the graph
unique_relationship = unique(ordered_relationship)
# have all unique actors into a vector, they will be the nodes in the graph
all_related_actors = unique(c(as.character(unique_relationship[,1]), as.character(unique_relationship[,2])))


# plot the relationship
library(igraph)
# actors = c(Boxer_id, boxer_all_fellows)
# length(actors)

gr = graph.data.frame(unique_relationship,  directed = FALSE, vertices = all_related_actors)  

# count the actors' appearence frequency in the relationships
cou = function(name) {
  count = sum(unique_relationship[,1] == name) + sum(unique_relationship[,2] == name)
  c(name, count)
}
# set node size
grade = data.frame(t(sapply(all_related_actors, cou)))
grade$size = ifelse(as.integer(as.character((grade$X2)))>1, log(as.integer(as.character((grade$X2))))*2, 1)
node.size = setNames(grade[,3], as.character(grade[,1]))
# set label size
label_size = ifelse(as.integer(as.character((grade$X2)))>35, 1, 0.01)
label.size = setNames(label_size, as.character(grade[,1]))

# find the gender for a given actor id
findGender = function(actor) {
  gender = subset(name, name == actor)$gender[1]
  c(actor, gender)
}

all_actor_gender = data.frame(t(sapply(all_related_actors, findGender)))
all_male_actor = as.character(subset(all_actor_gender, as.character(X2) == 'm')[,1])
V(gr)$color = ifelse(V(gr)$name == Boxer_name, 'blue', 
                     ifelse(V(gr)$name %in% all_male_actor, 'green', 'red'))

par(mar=c(0,0,1,0))
plot(gr, layout=layout.fruchterman.reingold, main="Boxer John's movies network", 
     vertex.label.dist=0.5, vertex.frame.color='grey', vertex.label.color='black',
     vertex.label.font=0.5, vertex.label.cex=label.size, vertex.size = node.size) # layout.fruchterman.reingold