'''
Make all mesh defined in Sipp & Lebedev (2007)
Used for mesh convergence on the cylinder case
(One might use a coarser mesh at some point though)
'''

# Mesh  xinfa  xinf  yinf  xplus  n1  n2  n3  :  nt(k)  ndof(k)
# C1    -60    200   30    1.5    24  7   1      191    864
# C2    -60    200   30    10     24  11  1      410    1853
# C3    -60    100   30    50     37  11  1      462    2090
# C4    -60    175   30    1.5    24  7   1      172    779
# C5    -30    200   30    1.5    24  7   1      167    757
# C6    -60    200   25    1.5    24  7   1      174    788

import gmsh_generate_cylinder as gm

if __name__=='__main__':
    C1 = [-60, 200, 30, 1.5, 24, 7,  1] 
    C2 = [-60, 200, 30, 10,  24, 11, 1] 
    C3 = [-60, 100, 30, 50,  37, 11, 1] 
    C4 = [-60, 175, 30, 1.5, 24, 7,  1] 
    C5 = [-30, 200, 30, 1.5, 24, 7,  1] 
    C6 = [-60, 200, 25, 1.5, 24, 7,  1] 
    C = [C1, C2, C3, C4, C5, C6]

    def make_mesh_dict(C):
        '''Make mesh dictionary for using gmsh_generate_cylinder'''
        fields = ['xinfa', 'xinf', 'yinf', 'xplus',
                  'n1', 'n2', 'n3']
        mesh_dict = {}
        for i in range(len(fields)):
            mesh_dict[fields[i]] = C[i]
        return mesh_dict
    
    for i, c in enumerate(C):
        mesh_dict = make_mesh_dict(c)
        mesh_dict['filename'] = './results/C'+str(i+1)
        gm.make_all(**mesh_dict)#, filename='./results/'+'C'+str(i+1))









