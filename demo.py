# from model.MyNet import *

# x = torch.rand(1,3,640,640).cuda()
# net = Net().cuda()
# out = net(x)
# print(out.shape)


# from Eval.eval import SOD_Eval
# # SOD_Eval(60, 'SOD_DataSet/EORSSD_aug', '/mnt/Disk1/WIT/cxh/Net_MSA_Decoder')
# SOD_Eval(60, 'SOD_DataSet/ORSSD_aug', '/mnt/Disk1/WIT/cxh/Net_MSA_Decoder')


from model.MyNet import Net
import torch
# if __name__ == '__main__':
#     input = torch.rand(2, 3, 352, 352).cuda()

#     model = Net().cuda()
#     output = model(input)[-2]
#     print(output.shape)
if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table
    from torchsummary import summary
    device = 'cuda'
    model = Net().cuda()
    model.eval()
    inputs = torch.randn(1, 3, 256, 256).to(device)
    output = model(inputs)
    print('模型性能测试')
    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, inputs)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")
    print(flop_count_table(flops, max_depth=1))