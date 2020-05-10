DA_folder = 'mmd-0.8-lr-0.0001-bz-400-iter-50000-scr-True-shar-True-fc-128-bn-False-tclf-0.0-sclf-1.0-tlabels-0-vclf-1-total-val-100';
CNN_folder = 'cnn-4-bn-True-trn-85000-bz-400-lr-1e-05-Adam-5.0k';
source_folder = 'cnn-4-bn-False-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k';

DA_stats = load(fullfile(DA_folder, 'roc_fit_output.txt'));
CNN_stats = load(fullfile(CNN_folder, 'roc_fit_output.txt'));
source_stats = load(fullfile(source_folder, 'roc_fit_output.txt'));

DA_AUC = load(fullfile(DA_folder, 'fit_auc.txt'));
CNN_AUC = load(fullfile(CNN_folder, 'fit_auc.txt'));
soruce_AUC = load(fullfile(source_folder, 'fit_auc.txt'));

figure(1)
linewidth = 2.5;
FontSize = 16;
plot(DA_stats(:,1), DA_stats(:,2), 'b-','linewidth', linewidth);
hold on;
plot(CNN_stats(:,1), CNN_stats(:,2), 'r--', 'linewidth', linewidth);
hold on;
plot(source_stats(:,1), source_stats(:,2), 'k--', 'linewidth', linewidth);
hold on;

xlabel('FPF');
ylabel('TPF')
legend(['AUC_{MDA}:',num2str(DA_AUC(1)),'\pm',num2str(DA_AUC(2))],...
     ['AUC_{170K}:',num2str(CNN_AUC(1)),'\pm',num2str(CNN_AUC(2))],...
     ['AUC_{SO}:',num2str(soruce_AUC(1)),'\pm',num2str(soruce_AUC(2))])   
set(findall(gcf,'-property','FontSize'),'FontSize',FontSize)
