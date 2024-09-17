


N_iter
            for (int k_gemm1_o = 0; k_gemm1_o < NI / KI; k_gemm1_o += 1)
            { // GEMM0是对K迭代。GEMM1就是对N迭代。但是还是按照KI和KII的尺寸来。我们此处无法迭代到底。只能迭代NI/KI次。是根据GEMM0的中间结果尺寸决定的。
                int k_gemm1_o_iter = N_iter*(NI/KI)+k_gemm1_o;
                // loadSmemA(SA, A, M, K, k_gemm1_o); // 不需要了。就是上一步的SC
                loadSmemB_new(S_gemm1_weight, gemm1_weight, T, N, k_gemm1_o_iter, T_iter);  // 形状是N*T，所以此处写成T, N
                __syncthreads();
                if(blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0&&k_gemm1_o==0){
                    printf("loadSmemB_new\n");
                    for(int kk=0;kk<128;kk++){
                        printf("%f  ", float(S_gemm1_weight[kk]));
                    }
                    printf("\n");
                }
                for (int ki = 0; ki < KI / KII; ki += 1)
                { // KI/KII=32/16=2
                    // 64x64x16 mma for each warp
                    loadFragA_new(FragA, SC, k_gemm1_o*KI+ki*KII); // 和之前的loadFragA不同，这里传入的第三个参量是当前真实的col位置，以及，这里要用k_gemm1_o而不是k_gemm1_o_iter，因为是在SC内部而不是去外部
                    loadFragB(FragB, S_gemm1_weight, ki);
                    for (int mii = 0; mii < MII / wmmaM; mii += 1)
                    {
                        for (int nii = 0; nii < NII / wmmaN; nii += 1)
                        {
                            // 16x16x16 for each wmma
                            nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
                        }
                    }
                }
            }



            for (int ko = 0; ko < K / KI; ko += 1)
            { // GEMM0是对K迭代。GEMM1就是对N迭代。
                // loadSmemA(SA, A, M, K, ko); // 不需要了。就是上一步的SC
                loadSmemB_new(S_gemm1_weight, gemm1_weight, T, N, ko, T_iter);  // 形状是N*T，所以此处写成T, N
                __syncthreads();
                if(blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0&&ko==0){
                    printf("loadSmemB_new\n");
                    for(int kk=0;kk<128;kk++){
                        printf("%f  ", float(S_gemm1_weight[kk]));
                    }
                    printf("\n");
                }
                for (int ki = 0; ki < KI / KII; ki += 1)
                { // KI/KII=32/16=2
                    // 64x64x16 mma for each warp
                    loadFragA_new(FragA, SC, ko*KI+ki*KII); // 和之前的loadFragA不同，这里传入的第三个参量是当前真实的col位置
                    loadFragB(FragB, S_gemm1_weight, ki);
                    for (int mii = 0; mii < MII / wmmaM; mii += 1)
                    {
                        for (int nii = 0; nii < NII / wmmaN; nii += 1)
                        {
                            // 16x16x16 for each wmma
                            nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
                        }
                    }
                }
            }
