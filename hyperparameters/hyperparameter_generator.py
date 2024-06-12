import argparse
import json


def comma_seperated_string_to_list(comma_seperated_String,dtype):
        if comma_seperated_String is None:
                return []
        return [dtype(x) for x in comma_seperated_String.split(',')]
    

def main():
    parser = argparse.ArgumentParser(description='Script Configuration via Command Line')
    parser.add_argument('--hyperparameters_file', type=str, default='hyperparameters/hyperparameters.json', help='String')
    parser.add_argument('--architecture', type=str, default='Simple', help='Float')
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--gen_lr',type=float,default=0.0002)
    parser.add_argument('--disc_lr',type=float,default=0.0002)
    parser.add_argument('--gen_optim', type=str, default='Adam', help='selected from https://pytorch.org/docs/stable/optim.html#algorithms, provided as a string')
    parser.add_argument('--disc_optim', type=str, default='Adam', help='selected from https://pytorch.org/docs/stable/optim.html#algorithms, provided as a string')
    parser.add_argument('--loss_fn', type=str, default='BCELoss', help='selected from https://pytorch.org/docs/stable/nn.html#loss-functions, provided as a string, without the nn')
    parser.add_argument('--save_frequency', type=int, default=10, help='comma seperated string of integers')
    
    args  =parser.parse_args()
    hyperparameters ={
        'architecture':args.architecture,
        'epochs':args.epochs,
        'batch_size':args.batch_size,
        'gen_lr':args.gen_lr,
        'disc_lr':args.disc_lr,
        'gen_optim':args.gen_optim,
        'disc_optim':args.disc_optim,
        'loss_fn':args.loss_fn
        
    }
    if args.architecture == "Simple":
        hyperparameters['noise_shape'] =(128,4,4,1)
   
      
    json_object = json.dumps(hyperparameters,indent=4) ### this saves the array in .json format)
    
    with open(args.hyperparameters_file, "w") as outfile:
            outfile.write(json_object)
        

    return 
if __name__ =="__main__":
    main()