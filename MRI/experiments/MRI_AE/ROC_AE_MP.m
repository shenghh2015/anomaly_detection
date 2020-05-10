% model_folder = 'AE1-MRI-cn-4-fr-32-ks-5-bn-True-skp-False-res-False-lr-0.0001-stps-300000-bz-50-tr-65k-vl-200-test-200-n-50.0';
model_folder = 'AE2-MRI-cn-4-fr-32-ks-3-bn-True-skp-False-res-False-lr-0.0001-stps-200000-bz-50-tr-65k-vl-200-test-200-n-40.0';

AE_stats = load(fullfile(model_folder, 'AE_roc_fit_output.txt'));
pixel_stats = load(fullfile(model_folder, 'pixel_roc_fit_output.txt'));

AE_AUC = load(fullfile(model_folder, 'AE_roc_fit_auc.txt'));
pixel_AUC = load(fullfile(model_folder, 'pixel_roc_fit_auc.txt'));

figure(1)
linewidth = 2.5;
FontSize = 16;
plot(AE_stats(:,1), AE_stats(:,2), 'b-','linewidth', linewidth);
hold on;
plot(pixel_stats(:,1), pixel_stats(:,2), 'r--', 'linewidth', linewidth);
hold on;

xlabel('FPF');
ylabel('TPF');
legend(['AUC_{AE}:',num2str(AE_AUC(1)),'\pm',num2str(AE_AUC(2))],...
     ['AUC_{MP}:',num2str(pixel_AUC(1)),'\pm',num2str(pixel_AUC(2))])   
set(findall(gcf,'-property','FontSize'),'FontSize',FontSize)
