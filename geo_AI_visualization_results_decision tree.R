# Visualization of the results of the decision tree classifier
# input from geo_AI_decision tree.py
# ML model by Rob van Putten
# Visualization by Nynke ter Heide

# bibliotheken inlezen
library(tidyverse)

#import data
pythondata <- read_csv("output_python_geoAI_tree.csv")

#make vector of GEF names
GEFs = unique(pythondata$GEFname)

# make plot for all GEF's
pythondata %>% 
  select(soilname, pred_soilname, depth, GEFname) %>% 
  rename(prediction = pred_soilname, real = soilname) %>% 
  gather("real_or_pred", "soil", c(real, prediction)) %>%
  filter(!is.na(soil)) %>% 
  ggplot(aes(real_or_pred, depth, color = soil)) +
  geom_point(shape = 15) +
  facet_wrap(~ GEFname, nrow = 1) +
  scale_color_manual(values = c("#000080", "#002aff", "#00d4ff", "#7bff7b", "#ffe600", "#ff4800", "#800000")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.3), axis.title.x = element_blank(), plot.title = element_text(hjust = 0.5)) +
  ggsave("all_GEFs3.jpg", dpi = 300, height = 8, width = 10)

# unique soil names
soils <- unique(pythondata$pred_soilname)
soils <- soils[!is.na(soils)]

# make confusion matrix of results from Python
cm <- pythondata %>% 
  select(pred_soilname, soilname) %>% 
  group_by(soilname, pred_soilname) %>% 
  summarise(pred_count = n()) %>% 
  ungroup() %>% 
  group_by(soilname) %>% 
  mutate(total = sum(pred_count)) %>% 
  group_by(soilname, pred_soilname) %>% 
  mutate(ratio = pred_count / total) %>% 
  select(soilname, pred_soilname, pred_count, ratio) %>% 
  mutate(correct = if_else(soilname == pred_soilname, "Yes", "No"),
         correct = if_else(is.na(correct), "No", correct)) %>% 
  ungroup()

# add empty values to confusion matrix 
cm_empty <- data.frame(matrix(ncol=length(soils), nrow=length(soils)))
names(cm_empty) <- soils
cm_empty$soilname <- soils
cm_empty <- gather(cm_empty, key="pred_soilname", value = "ratio", -soilname)
cm_empty$ratio <- as.numeric(0)
cm_empty$pred_count <- as.numeric(0)
cm_empty$correct <- "zero"
cm_empty <- anti_join(cm_empty, cm, by = c("pred_soilname", "soilname"))
cm <- rbind(cm, cm_empty)

# plot confusion matrix
ggplot(cm, aes(pred_soilname, soilname)) + 
  geom_tile(aes(fill = ratio), colour = "grey") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank()) +
  geom_text(aes(label = pred_count, color = correct)) +
  scale_fill_continuous(high = 1, low = 0.00001, na.value = 'white') +
  labs(title = "Confusion matrix",
       subtitle = "Number of predictions for each real soil class") +
  xlab("predicted soil") +
  ylab("real soil") +
  ggsave("cm_numbers3.jpg", height = 4, width = 5)

# plot confusion matrix
ggplot(cm, aes(pred_soilname, soilname)) + 
  geom_tile(aes(fill = ratio), colour = "grey") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank()) +
  geom_text(aes(label = round(ratio,2), color = correct)) +
  scale_fill_continuous(high = 1, low = 0.00001, na.value = 'white') +
  labs(title = "Confusion matrix",
       subtitle = "Ratio of predictions for each real soil class") +
  xlab("predicted soil") +
  ylab("real soil") +
  ggsave("cm_ratio3.jpg", height = 4, width = 5)



