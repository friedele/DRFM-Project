load("acc.mat");
load("valAcc.mat");

plot(smoothdata(acc*100),LineWidth=1.5);
hold on;
plot(smoothdata(valAcc*100),LineWidth=1.5);
grid on;
legend('Trained Data','Validadtion Data','Location','northwest')
ylim([40 102])
ylabel('Accuracy (%)')
xlabel({'Epochs';'(images=4000/batchSize=16: 250 iterations per Epoch)'}, 'FontSize',10);
title('Model Relative Error')