relPath="../modules/"

for configRoot in "./gene/SplitMerge/" "./gene/Gibbs/"
do
    configPath=$configRoot"config.json"
    python -u $relPath"experiment.py" --configPath $configPath --overwrite
done

for configRoot in "./kRegular/"
do
    configPath=$configRoot"config.json"
    python -u $relPath"experiment.py" --configPath $configPath --overwrite
done

for configRoot in "./synthetic_N=500/fixedParams/" "./synthetic_N=500/variedParams/"
do
    configPath=$configRoot"config.json"
    python -u $relPath"experiment.py" --configPath $configPath --overwrite
done