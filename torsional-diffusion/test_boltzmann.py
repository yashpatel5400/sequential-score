import tqdm, yaml, pickle, json
from argparse import ArgumentParser, Namespace

from utils.utils import get_model, prune_conformers, get_conformer_energies
from utils.boltzmann import *

parser = ArgumentParser()
parser.add_argument('--model_dir', type=str, default=None, help='')
parser.add_argument('--original_model_dir', type=str, default=None, help='')
parser.add_argument('--ckpt', type=str, default='best_model.pt', help='')
parser.add_argument('--temp', type=float, default=300, help='')
parser.add_argument('--ais_steps', type=int, default=0, help='')
parser.add_argument('--mcmc_sigma', type=float, default=0.1, help='')
parser.add_argument('--n_samples', type=int, default=32, help='')
parser.add_argument('--model_steps', type=int, default=5, help='')
parser.add_argument('--test_pkl', type=str, default='data/DRUGS/test_mols.pkl', help='')
parser.add_argument('--out', type=str, default='boltzmann.out', help='')
args = parser.parse_args()

"""
    Evaluates the ESS given a trained torsional Boltzmann generator 
"""

test_smiles = ["[H]C([H])([H])C([H])([H])C([H])([H])C([H])(C([H])([H])C([H])([H])C([H])(C([H])([H])[H])C([H])([H])C([H])([H])[H])C([H])([H])C([H])(C([H])([H])C([H])([H])[H])C([H])(C([H])([H])C([H])([H])[H])C([H])([H])C([H])([H])[H]"]
standard = 18.9667232606877
total = 12.107139015589842
mols = [Chem.AddHs(Chem.MolFromSmiles(smile)) for smile in test_smiles]
unique_symbols = np.unique([[atom.GetSymbol() for atom in mol.GetAtoms()] for mol in mols])
types = dict(zip(unique_symbols, range(len(unique_symbols)))) # map of atoms -> index

test_data = []
for test_smi in test_smiles:
    mol = Chem.AddHs(Chem.MolFromSmiles(test_smi))
    AllChem.EmbedMolecule(mol)
    data = featurize_mol(mol, types)
    data.mol = mol
    data.edge_mask, data.mask_rotate = get_transformation_mask(data)
    data.edge_mask = torch.tensor(data.edge_mask)
    # if data.mask_rotate.shape[0] < 3 or data.mask_rotate.shape[0] > 7: continue
    data.pos = [mol.GetConformers()[0].GetPositions()]
    test_data.append(data)
print('Testing on', len(test_data), 'molecules')

if not args.model_dir:
    resampler = BaselineResampler(ais_steps=args.ais_steps, temp=args.temp,
                                  mcmc_sigma=args.mcmc_sigma, n_samples=args.n_samples)
else:
    args2 = {}
    if args.original_model_dir:
        # load the model with the original model parameters
        with open(f'{args.original_model_dir}/model_parameters.yml') as f:
            args2.update(yaml.full_load(f))
    else:
        with open(f'{args.model_dir}/model_parameters.yml') as f:
            args2.update(yaml.full_load(f))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(Namespace(**args2))
    state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if args.original_model_dir:
        with open(f'{args.model_dir}/model_parameters.yml') as f:
            args2.update(yaml.full_load(f))
    args2 = Namespace(**args2)
    print("Original model: temp", args2.temp if "temp" in args2 else "-",
          " steps", args2.boltzmann_steps if "boltzmann_steps" in args2 else "-")
    print("Current settings: temp", args.temp, " steps", args.model_steps,
          " sigma min", args2.sigma_min, " sigma max", args2.sigma_max)
    args2.boltzmann_steps = args.model_steps
    args2.temp = args.temp
    args2.boltzmann_confs = args.n_samples

    resampler = BoltzmannResampler(args=args2, model=model)

ess = []
for mol in tqdm.tqdm(test_data):
    ess_ = resampler.resample(mol)
    ess.append(ess_)
    
    AllChem.MMFFOptimizeMoleculeConfs(mol.mol)
    pruned_mol = prune_conformers(mol.mol, tfd_thresh=0.05)
    energies = get_conformer_energies(pruned_mol)
    gibbs_reward = np.sum(np.exp(-1.0 * (np.array(energies) - standard)) / total)

print('mean', np.mean(ess), 'median', np.median(ess), 'gibbs', gibbs_reward)

with open(args.out, 'a') as f:
    f.write(json.dumps({
        **args.__dict__,
        'ess': ess
    }) + '\n')
