library(rjson)
library(foreach)
library(iterators)
library(rlist)
library(data.table)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(grid)

result_usup  <- fromJSON(file = "output_large_unsupervised.json")

result_sup <- fromJSON(file = "output_large_supervised.json")

result_map <- data.table(map.data = result_sup$map$data)

result_wins <- data.table(map.wins = result_sup$tag_activation_map$data)

uresult_map <- data.table(map.data = result_usup$map$data)

#tlist <- seq(0, 765, by = 3)
tlist <- seq(0, 1197, by = 3)
tagwins.real  <- foreach(i=iter(tlist)) %do% {
  x <- if(grep("positive", names(result_sup$classes)) == 1) {
    i + 1
  } else if(grep("positive", names(result_sup$classes)) == 2) {
    i + 2
  } else {
    i + 3
  }
  output.tagwins.real <- result_wins$map.wins[x]

}

tagwins.fake  <- foreach(i=iter(tlist)) %do% {
  y <- if(grep("negative", names(result_sup$classes)) == 1) {
    i + 1
  } else if(grep("negative", names(result_sup$classes)) == 2) {
    i + 2
  } else {
    i + 3
  }
  output.tagwins.fake <- result_wins$map.wins[y]
}

tagwins.middleman  <- foreach(i=iter(tlist)) %do% {
  y <- if(grep("middleman", names(result_sup$classes)) == 1) {
    i + 1
  } else if(grep("middleman", names(result_sup$classes)) == 2) {
    i + 2
  } else {
    i + 3
  }
  output.tagwins.fake <- result_wins$map.wins[y]
}

clmn <- c(rep(1, 12), rep(2, 12), rep(3, 12),
                      rep(4, 12), rep(5, 12), rep(6, 12),
                      rep(7, 12), rep(8, 12), rep(9, 12), rep(10, 12),
                      rep(11, 12), rep(12, 12))
r <- rep(seq(1, 12, by = 1), 12)


clmn <- c(rep(1, 16), rep(2, 16), rep(3, 16),
                      rep(4, 16), rep(5, 16), rep(6, 16),
                      rep(7, 16), rep(8, 16), rep(9, 16), rep(10, 16),
                      rep(11, 16), rep(12, 16), rep(13, 16), rep(14, 16),
                      rep(15, 16), rep(16, 16))
r <- rep(seq(1, 16, by = 1), 16)

clmn <- c(rep(1, 20), rep(2, 20), rep(3, 20),
                      rep(4, 20), rep(5, 20), rep(6, 20),
                      rep(7, 20), rep(8, 20), rep(9, 20), rep(10, 20),
                      rep(11, 20), rep(12, 20), rep(13, 20), rep(14, 20),
                      rep(15, 20), rep(16, 20),
                      rep(17, 20), rep(18, 20), rep(19, 20), rep(20, 20))
r <- rep(seq(1, 20, by = 1), 20)



pper <- foreach(i = iter(seq(1, 400, by = 1))) %do% {
  tr <- tagwins.real[[i]]
  res <- tr / result_sup$tag_activation_map$data[i] * 100
}

nper <- foreach(i = iter(seq(1, 400, by = 1))) %do% {
  tr <- tagwins.middleman[[i]]
  res <- tr / result_sup$tag_activation_map$data[i] * 100
}

mper <- foreach(i = iter(seq(1, 400, by = 1))) %do% {
  tr <- tagwins.middleman[[i]]
  res <- tr / result_sup$tag_activation_map$data[i] * 100
}

pper <- unlist(pper, recursive = FALSE)
nper <- unlist(nper, recursive = FALSE)
mper <- unlist(mper, recursive = FALSE)

results2 <- data.table(
                     column = clmn,
                     row = r,
                     activation.map = result_usup$activation_map$data,
                     tag.map = result_usup$tag_map$data
)

results <- data.table(
                     column = clmn,
                     row = r,
                     tag.map = result_sup$tag_map$data,
                     tag.wins.real = tagwins.real,
                     tag.wins.fake = tagwins.fake,
                     activation.map = pper,
                     total.wins = result_sup$activation_map$data
)

results3 <- data.table(
                     column = clmn,
                     row = r,
                     tag.map = result_sup$tag_map$data,
                     tag.wins.real = tagwins.real,
                     tag.wins.fake = tagwins.fake,
                     activation.map = mper,
                     total.wins = result_sup$activation_map$data
)

hm10_usup <- function(res, name) {
  pdf(name, paper = "legal")
  plot1 <- ggplot(data = res, aes(x = column, y = row, fill = activation.map)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(high = "#1B9E77", low = "#D95F02", mid = "white",
                               midpoint = 100, space = "Lab",
                               name = "Activations") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                        size = 12, hjust = 1)) +
    geom_text(aes(label = tag.map), size = 1) +
    coord_fixed()

  print(plot1)
  dev.off()
}
#limit = c(r[1], r[2]),
hm10 <- function(res, name) {
  pdf(name, paper = "legal")
  plot1 <- ggplot(data = res,
                  aes(x = column, y = row, group = tag.map, fill = tag.map)) +
    geom_tile(color = "white") +
    #scale_fill_gradient2(high = "#1B9E77", low = "#D95F02", mid = "white",
    #                           midpoint = 50,  space = "Lab",
    #                           name = "Win Percentage \nReal") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                        size = 12, hjust = 1)) +
    geom_text(aes(label = tag.map), size = 1) +
    coord_fixed()

  print(plot1)
  dev.off()
}

hm10(results, "som15_supervised.pdf")
hm10(results3, "som15_supervised_mm.pdf")
hm10_usup(results2, "som15_unsupervised.pdf")
