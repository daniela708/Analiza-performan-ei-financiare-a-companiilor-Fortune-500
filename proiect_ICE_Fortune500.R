
# 1. BAZA DE DATE

# Citirea datelor

date <- read.csv("fortune500_proiect.csv", stringsAsFactors = FALSE)

cat("=== STRUCTURA SETULUI DE DATE ===\n")
str(date)

cat("\n=== PRIMELE 6 RANDURI ===\n")
head(date)

cat("\n=== STATISTICI DESCRIPTIVE ===\n")
summary(date[, c("venituri_mil","profit_mil","angajati",
                  "active_mil","marja_profit_pct")])

# Descrierea atributelor:

cat("\n=== DISTRIBUTIA CLASELOR - profitabila ===\n")
print(table(date$profitabila))

cat("\n=== DISTRIBUTIA CLASELOR - segment_performanta ===\n")
print(table(date$segment_performanta))

cat("\n=== DISTRIBUTIA PE SECTOARE (top 10) ===\n")
print(sort(table(date$sector), decreasing = TRUE)[1:10])

#  Verificare valori lipsa

cat("\n=== VALORI LIPSA ===\n")
print(colSums(is.na(date)))


#  Standardizarea datelor 

vars_num <- c("venituri_mil","profit_mil","angajati",
              "active_mil","marja_profit_pct")
date_scaled <- as.data.frame(scale(date[, vars_num]))
cat("\n=== DATE STANDARDIZATE - primele 5 randuri ===\n")
print(round(head(date_scaled, 5), 3))


# Transformarea in variabile factor

date$sector              <- factor(date$sector)
date$profitabila         <- factor(date$profitabila, levels = c("Nu","Da"))
date$segment_performanta <- factor(date$segment_performanta,
                                   levels = c("Pierdere","Redusa",
                                              "Medie","Ridicata"))
# statistici descriptive
summary(date[, vars_num])

# 2. CLUSTERIZAREA FUZZY


library(e1071)
library(cluster)
library(factoextra)
library(ggplot2)
library(dplyr)

date_cl <- date_scaled[, c("venituri_mil","profit_mil",
                             "angajati","marja_profit_pct")]

set.seed(42)
# k=3 clustere: companii mici/medii/mari
# m=2 gradul de fuzzificare standard
model_cmeans <- cmeans(date_cl, centers = 3,
                        iter.max = 100, m = 2, method = "cmeans")

cat("\n=== CLUSTERIZARE FUZZY - cmeans ===\n")
cat("Numar clustere: k = 3\n")
cat("Grad de fuzzificare: m = 2\n")

cat("\n--- Centroizii clusterelor (spatiu standardizat) ---\n")
print(round(model_cmeans$centers, 4))

memb_df <- as.data.frame(model_cmeans$membership)
colnames(memb_df) <- c("Cl1_Mici","Cl2_Medii","Cl3_Mari")

cat("\n--- Grade de apartenenta - primele 10 observatii ---\n")
print(round(head(memb_df, 10), 4))

date$cluster_cm <- factor(model_cmeans$cluster)
cat("\n--- Distributia observatiilor pe clustere ---\n")
print(table(date$cluster_cm))

cat("\n--- Primele 8 companii din Clusterul 1 ---\n")
print(date[date$cluster_cm == 1,
           c("companie","venituri_mil","profit_mil","sector")][1:8,])
cat("\n--- Primele 8 companii din Clusterul 2 ---\n")
print(date[date$cluster_cm == 2,
           c("companie","venituri_mil","profit_mil","sector")][1:8,])
cat("\n--- Primele 8 companii din Clusterul 3 ---\n")
print(date[date$cluster_cm == 3,
           c("companie","venituri_mil","profit_mil","sector")][1:8,])

# Grafic in plan: venituri vs marja_profit
plot_cm <- data.frame(
  venituri   = date$venituri_mil,
  marja      = date$marja_profit_pct,
  Cluster    = factor(model_cmeans$cluster,
                      labels = c("Mici","Medii","Mari")),
  memb1      = memb_df$Cl1_Mici
)
ggplot(plot_cm, aes(x = venituri, y = marja, color = Cluster)) +
  geom_point(aes(size = memb1), alpha = 0.65) +
  scale_color_manual(values = c("#E74C3C","#27AE60","#2980B9")) +
  scale_x_log10(labels = scales::comma) +
  scale_size_continuous(name = "Apartinere Cluster 1") +
  labs(title    = "Clusterizare Fuzzy cmeans - Fortune 500",
       subtitle = "Dimensiunea punctului = grad de apartenenta la Cl.1 (Companii mici)",
       x = "Venituri (mil. USD, scala log)",
       y = "Marja de profit (%)") +
  theme_minimal()


#  Functia fanny 

set.seed(42)
model_fanny <- fanny(date_cl, k = 3, memb.exp = 2, diss = FALSE)

cat("\n=== CLUSTERIZARE FUZZY - fanny ===\n")
cat("Numar clustere: k = 3\n")

fanny_memb <- as.data.frame(model_fanny$membership)
colnames(fanny_memb) <- c("Cluster_1","Cluster_2","Cluster_3")
cat("\n--- Grade de apartenenta - primele 10 observatii ---\n")
print(round(head(fanny_memb, 10), 4))

cat("\n--- Coeficientul de partitie Dunn ---\n")
cat(sprintf("  Nenormalizat: %.4f  (se compara cu 1/k = %.4f)\n",
            model_fanny$coeff[1], 1/3))
cat(sprintf("  Normalizat:   %.4f  (cat mai aproape de 0 = fuzzificare ridicata)\n",
            model_fanny$coeff[2]))

# Grafic factoextra
fviz_cluster(model_fanny,
             ellipse.type = "norm", repel = FALSE,
             palette      = c("#E74C3C","#27AE60","#2980B9"),
             ggtheme      = theme_minimal()) +
  labs(title = "FANNY Fuzzy Clustering - Fortune 500")

# Coeficienti silueta
sil <- silhouette(model_fanny$clustering, dist(date_cl))
cat(sprintf("\n--- Coeficient silueta mediu: %.4f ---\n", mean(sil[,3])))
cat("(>0.50 = clustere bine definite; 0.25-0.50 = structura acceptabila)\n")

fviz_silhouette(sil,
                palette = c("#E74C3C","#27AE60","#2980B9"),
                ggtheme = theme_minimal()) +
  labs(title = "Coeficienti Silueta - FANNY Clustering")


#  REGRESIA LOGISTICA

#install.packages("ROCR")
library(caTools)
library(ROCR)
library(nnet)

# Split 80/20 antrenare/testare
set.seed(123)
split_idx  <- sample.split(date$profitabila, SplitRatio = 0.80)
train_log  <- subset(date, split_idx == TRUE)
test_log   <- subset(date, split_idx == FALSE)
cat(sprintf("\nSet antrenare: %d obs. | Set testare: %d obs.\n",
            nrow(train_log), nrow(test_log)))


#  Regresia logistica binomiala

cat("\n=== REGRESIA LOGISTICA BINOMIALA ===\n")
cat("Variabila dependenta: profitabila\n")
cat("  'Nu' = profit net <= 0 (companie in pierdere)\n")
cat("  'Da' = profit net >  0 (companie profitabila)\n")
print(table(date$profitabila))

# Figura - distributia pe categorii
ggplot(date, aes(x = log10(venituri_mil),
                  y = marja_profit_pct,
                  color = profitabila)) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c("Nu" = "#E74C3C","Da" = "#27AE60")) +
  labs(title = "Profitabilitate companiilor Fortune 500",
       x     = "log10(Venituri mil. USD)",
       y     = "Marja de profit (%)") +
  theme_minimal()

# Modelul GLM
model_binom <- glm(profitabila ~ venituri_mil + angajati +
                     active_mil + marja_profit_pct + crst_venituri_pct,
                   data   = train_log,
                   family = binomial(link = "logit"))
cat("\n--- Rezumatul modelului GLM ---\n")
print(summary(model_binom))

cat("\n--- Ecuatia de regresie logistica binomiala ---\n")
cat("logit(p) = b0 + b1*venituri_mil + b2*angajati\n")
cat("         + b3*active_mil + b4*marja_profit_pct\n")
cat("         + b5*crst_venituri_pct\n")

cat("\n--- exp(coef(model)) - Odds Ratios ---\n")
or <- exp(coef(model_binom))
print(round(or, 6))

cat("\nInterpretare selectiva:\n")
cat(" marja_profit_pct:  O crestere cu 1% a marjei multiplica sansele cu",
    round(or["marja_profit_pct"], 4), "\n")
cat(" crst_venituri_pct: O crestere cu 1% a veniturilor multiplica sansele cu",
    round(or["crst_venituri_pct"], 4), "\n")

# Predictia
pred_prob_test  <- predict(model_binom, newdata = test_log,
                            type = "response")
pred_cls_test   <- factor(ifelse(pred_prob_test > 0.5,"Da","Nu"),
                           levels = c("Nu","Da"))
pred_prob_train <- predict(model_binom, newdata = train_log,
                            type = "response")
pred_cls_train  <- factor(ifelse(pred_prob_train > 0.5,"Da","Nu"),
                           levels = c("Nu","Da"))

cat("\n--- Matricea de confuzie - Set ANTRENARE ---\n")
print(table(Actual = train_log$profitabila,
            Predicted = pred_cls_train))
cat(sprintf("Rata acuratete antrenare: %.2f%%\n",
            mean(pred_cls_train == train_log$profitabila)*100))

cat("\n--- Matricea de confuzie - Set TESTARE ---\n")
print(table(Actual = test_log$profitabila,
            Predicted = pred_cls_test))
acc_binom <- mean(pred_cls_test == test_log$profitabila)
cat(sprintf("Rata acuratete testare: %.2f%%\n", acc_binom*100))

# Curba ROC + AUC
#pred_rocr <- prediction(pred_prob_test, as.numeric(test_log$profitabila) - 1)
#perf_rocr <- performance(pred_rocr, "tpr", "fpr")
#auc_binom <- as.numeric(performance(pred_rocr, "auc")@y.values)

perf_rocr <- performance(pred_rocr,"tpr","fpr")
auc_binom <- as.numeric(performance(pred_rocr,"auc")@y.values)
plot(perf_rocr, col = "blue", lwd = 2,
     main = "Curba ROC - Regresia Logistica Binomiala")
abline(0, 1, lty = 2, col = "gray")
legend("bottomright",
       legend = paste("AUC =", round(auc_binom, 4)),
       col = "blue", lwd = 2)
cat(sprintf("\nAUC = %.4f => modelul separa corect clasele in proportie de %.2f%%\n",
            auc_binom, auc_binom*100))


#  Regresia logistica multinomiala

cat("\n=== REGRESIA LOGISTICA MULTINOMIALA ===\n")
cat("Variabila dependenta: segment_performanta (4 clase)\n")
cat("  Pierdere: marja profit < 0%\n")
cat("  Redusa:   marja profit 0-5%\n")
cat("  Medie:    marja profit 5-15%\n")
cat("  Ridicata: marja profit > 15%\n")
print(table(date$segment_performanta))

date$segment_performanta <- relevel(date$segment_performanta,
                                     ref = "Pierdere")
train_multi <- subset(date, split_idx == TRUE)
test_multi  <- subset(date, split_idx == FALSE)

model_multi <- multinom(
  segment_performanta ~ venituri_mil + angajati +
    active_mil + crst_venituri_pct,
  data  = train_multi, trace = FALSE)

cat("\n--- Rezumatul modelului multinomial ---\n")
print(summary(model_multi))

cat("\n--- Ecuatiile de regresie (log-odds fata de Pierdere) ---\n")
cat("log(P(Redusa)/P(Pierdere))   = b0_R + b1_R*venituri + ...\n")
cat("log(P(Medie)/P(Pierdere))    = b0_M + b1_M*venituri + ...\n")
cat("log(P(Ridicata)/P(Pierdere)) = b0_Ri + b1_Ri*venituri + ...\n")

cat("\n--- exp(coef) - Odds Ratios ---\n")
print(round(exp(coef(model_multi)), 4))

pred_multi <- predict(model_multi, newdata = test_multi)
cm_multi   <- table(Actual = test_multi$segment_performanta,
                     Predicted = pred_multi)
cat("\n--- Matricea de confuzie - Model Multinomial ---\n")
print(cm_multi)
acc_multi <- mean(pred_multi == test_multi$segment_performanta)
cat(sprintf("Rata acuratete: %.2f%%\n", acc_multi*100))


#  ARBORI DE DECIZIE


library(rpart)
library(rpart.plot)
library(tree)
library(caret)
library(pROC)

set.seed(123)
split2  <- createDataPartition(y = date$profitabila,
                                p = 0.60, list = FALSE)
train_t <- date[split2, ]
test_t  <- date[-split2, ]


#  Arbore de clasificare (target: profitabila)

#install.packages("rpart")
#install.packages("rpart.plot")

library(rpart)
library(rpart.plot)
cat("\n=== ARBORE DE CLASIFICARE - profitabila ===\n")
library(caret)

set.seed(123)
split2  <- createDataPartition(y = date$profitabila, p = 0.60, list = FALSE)
train_t <- date[split2, ]
test_t  <- date[-split2, ]
cat(paste("Arbori - Set antrenare:", nrow(train_t), "| Set testare:", nrow(test_t)))
arbore_cls <- rpart(
  profitabila ~ venituri_mil + profit_mil + angajati +
    active_mil + marja_profit_pct + crst_venituri_pct,
  data   = train_t,
  method = "class")
#install.packages("caret")
library(caret)

rpart.plot(arbore_cls, extra = 106,
           main = "Arbore de clasificare - Profitabila (Da/Nu)")
print(arbore_cls)

pred_cls_arb  <- predict(arbore_cls, test_t, type = "class")
conf_cls      <- table(test_t$profitabila, pred_cls_arb)
cat("\n--- Matricea de confuzie - Arbore clasificare ---\n")
print(conf_cls)
err_cls <- mean(pred_cls_arb != test_t$profitabila)
cat(sprintf("Rata eroare: %.2f%% | Acuratete: %.2f%%\n",
            err_cls*100, (1-err_cls)*100))
#install.packages("pROC")
library(pROC)
# ROC arbore clasificare
pred_prob_arb <- predict(arbore_cls, test_t, type = "prob")
roc_arb <- roc(test_t$profitabila, pred_prob_arb[,"Da"])
plot(roc_arb, col = "darkgreen",
     main = "Curba ROC - Arbore de clasificare")
cat(sprintf("AUC arbore clasificare: %.4f\n", auc(roc_arb)))

library(rpart)
library(rpart.plot)
# Curatarea arborelui
cat("\n--- Curatarea arborelui (Pruning) ---\n")
plotcp(arbore_cls)
printcp(arbore_cls)
cp_opt <- arbore_cls$cptable[
  which.min(arbore_cls$cptable[,"xerror"]),"CP"]
cat(sprintf("CP optim (min xerror): %.6f\n", cp_opt))

arbore_curat <- prune(arbore_cls, cp = cp_opt)
rpart.plot(arbore_curat, extra = 106,
           main = "Arbore clasificare CURATAT - Profitabila")

pred_curat <- predict(arbore_curat, test_t, type = "class")
cat("\n--- Matricea de confuzie - Arbore curatat ---\n")
print(table(test_t$profitabila, pred_curat))
cat(sprintf("Acuratete arbore curatat: %.2f%%\n",
            mean(pred_curat == test_t$profitabila)*100))


#  Arbore de regresie (target: marja_profit_pct)

#install.packages("tree")
library(tree)
cat("\n=== ARBORE DE REGRESIE - marja_profit_pct ===\n")

arbore_reg <- tree(
  marja_profit_pct ~ venituri_mil + angajati +
    active_mil + crst_venituri_pct,
  data = train_t)
plot(arbore_reg)
text(arbore_reg, pretty = 0)
title("Arbore de regresie - Marja de profit (%)")

set.seed(12)
cv_reg       <- cv.tree(arbore_reg)
best_size    <- cv_reg$size[which.min(cv_reg$dev)]
cat(sprintf("Marimea optima arbore (noduri terminale): %d\n", best_size))
plot(cv_reg$size, cv_reg$dev, type = "b",
     xlab = "Noduri terminale", ylab = "Devianța (CV)",
     main = "Cross-Validation - Arbore regresie")

arbore_reg_curat <- prune.tree(arbore_reg, best = best_size)
plot(arbore_reg_curat)
text(arbore_reg_curat, pretty = 0)
title("Arbore de regresie CURATAT - Marja de profit (%)")

pred_reg <- predict(arbore_reg_curat, test_t)
mse_reg  <- mean((pred_reg - test_t$marja_profit_pct)^2)
cat(sprintf("MSE arbore regresie curatat: %.4f\n", mse_reg))

plot(pred_reg, test_t$marja_profit_pct,
     xlab = "Valori previzionate (%)",
     ylab = "Valori reale (%)",
     main = "Predictie vs Realitate - Arbore regresie",
     col  = "darkgreen", pch = 16)
abline(0, 1, col = "red", lwd = 2)


#  KNN - CLASIFICARE


library(class)

cat("\n=== KNN - CLASIFICARE profitabila ===\n")

vars_knn  <- c("venituri_mil","profit_mil","angajati",
               "active_mil","marja_profit_pct","crst_venituri_pct")
X_all     <- scale(date[, vars_knn])
Y_all     <- date$profitabila

set.seed(123)
idx_knn     <- sample(1:nrow(X_all), size = 0.70*nrow(X_all))
X_train_knn <- X_all[idx_knn,]
X_test_knn  <- X_all[-idx_knn,]
Y_train_knn <- Y_all[idx_knn]
Y_test_knn  <- Y_all[-idx_knn]

cat(sprintf("Set antrenare KNN: %d obs.\n", length(Y_train_knn)))
cat(sprintf("Set testare KNN:   %d obs.\n", length(Y_test_knn)))

# KNN cu k=5
knn_pred <- knn(train = X_train_knn, test = X_test_knn,
                cl = Y_train_knn, k = 5, prob = TRUE)

cat("\n--- Matricea de confuzie KNN (k=5) ---\n")
cm_knn <- table(Predictie = knn_pred, Realitate = Y_test_knn)
print(cm_knn)
acc_knn <- mean(knn_pred == Y_test_knn)
cat(sprintf("Rata acuratete KNN (k=5): %.2f%%\n", acc_knn*100))

# Curba ROC
library(pROC)
prob_knn <- ifelse(knn_pred == "Da",
                   attr(knn_pred,"prob"),
                   1 - attr(knn_pred,"prob"))
roc_knn  <- roc(response  = Y_test_knn,
                predictor = prob_knn,
                levels    = c("Nu","Da"))
plot(roc_knn, col = "purple", lwd = 2,
     main = "Curba ROC - KNN (k=5)")
auc_knn <- auc(roc_knn)
cat(sprintf("AUC KNN: %.4f\n", auc_knn))

# Alegerea optima a lui k
acc_vec <- numeric(15)
for(k in 1:15){
  knn_k      <- knn(X_train_knn, X_test_knn, Y_train_knn, k = k)
  acc_vec[k] <- mean(knn_k == Y_test_knn)
}
plot(1:15, acc_vec, type = "b", pch = 19, col = "purple",
     xlab = "Valoarea lui k", ylab = "Acuratete",
     main = "Acuratete KNN in functie de k")
abline(v = which.max(acc_vec), lty = 2, col = "red")
cat(sprintf("Valoarea optima k = %d, acuratete: %.2f%%\n",
            which.max(acc_vec), max(acc_vec)*100))


# RETELE NEURONALE - CLASIFICARE

#install.packages("neuralnet")

library(neuralnet)
library(caTools)
library(ROCR)

cat("\n=== RETEA NEURONALA - CLASIFICARE profitabila ===\n")

# Normalizare min-max in [0,1]
maxim     <- apply(date[, vars_knn], 2, max)
minim     <- apply(date[, vars_knn], 2, min)
date_norm <- as.data.frame(
  scale(date[, vars_knn], center = minim, scale = maxim - minim))
date_norm$profitabila_bin <- as.numeric(date$profitabila == "Da")

set.seed(123)
split_rn  <- sample.split(date_norm$profitabila_bin,
                           SplitRatio = 0.75)
train_rn  <- subset(date_norm, split_rn == TRUE)
test_rn   <- subset(date_norm, split_rn == FALSE)
cat(sprintf("Set antrenare RNA: %d obs.\n", nrow(train_rn)))
cat(sprintf("Set testare RNA:   %d obs.\n", nrow(test_rn)))

# Antrenare retea cu 5 neuroni in stratul ascuns
set.seed(42)
retea <- neuralnet(
  profitabila_bin ~ venituri_mil + profit_mil + angajati +
    active_mil + marja_profit_pct + crst_venituri_pct,
  data          = train_rn,
  hidden        = 5,
  linear.output = FALSE,
  threshold     = 0.05,
  stepmax       = 1e6,
  lifesign      = "minimal")

plot(retea, rep = "best")


# Predictia

pred_rn_val <- tail(as.vector(retea$net.result[[1]]), nrow(test_rn))
pred_rn_cls <- ifelse(pred_rn_val > 0.5, "Da", "Nu")
actual_rn   <- ifelse(test_rn$profitabila_bin == 1, "Da", "Nu")

tab_rn  <- table(Actual = actual_rn, Predicted = pred_rn_cls)
cat("\n--- Matricea de confuzie - Retea Neuronala ---\n")
print(tab_rn)
acc_rn <- sum(diag(tab_rn)) / sum(tab_rn)
cat(sprintf("Rata acuratete RNA: %.2f%%\n", acc_rn*100))

# ROC + AUC
# ROC + AUC
library(ROCR)
pred_rocr_rn <- ROCR::prediction(as.numeric(pred_rn_val),
                                 as.numeric(test_rn$profitabila_bin))
perf_rocr_rn <- ROCR::performance(pred_rocr_rn,"tpr","fpr")
auc_rn       <- as.numeric(ROCR::performance(pred_rocr_rn,"auc")@y.values)
plot(perf_rocr_rn, col = "darkorange", lwd = 2,
     main = "Curba ROC - Retea Neuronala")
abline(0, 1, lty = 2, col = "gray")
legend("bottomright",
       legend = paste("AUC =", round(auc_rn, 4)),
       col = "darkorange", lwd = 2)
cat(sprintf("AUC Retea Neuronala: %.4f\n", auc_rn))