
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func private @scale(%arg0 : tensor<1x4096x64xf32>, %arg1 : f32) -> tensor<1x4096x64xf32> {
    %empty = tensor.empty() : tensor<1x4096x64xf32>
    %generic = linalg.generic  {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<1x4096x64xf32>)
      outs(%empty : tensor<1x4096x64xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.mulf %b0, %arg1 : f32
        linalg.yield %0 : f32
      } -> tensor<1x4096x64xf32>

    return %generic : tensor<1x4096x64xf32>
}

func.func @main(
    %q : tensor<1x4096x64xf32>,
    %k : tensor<1x4096x64xf32>,
    %v : tensor<1x4096x64xf32>,
    %scale : tensor<f32>,
    %qscale : tensor<f32>,
    %kscale : tensor<f32>,
    %vscale : tensor<f32>) -> tensor<1x4096x64xf32> {

    %scalef32 = tensor.extract %scale[] : tensor<f32>
    %qscalef32 = tensor.extract %qscale[] : tensor<f32>
    %kscalef32 = tensor.extract %kscale[] : tensor<f32>
    %vscalef32 = tensor.extract %vscale[] : tensor<f32>

    %qf8 = arith.truncf %q : tensor<1x4096x64xf32> to tensor<1x4096x64xf8E4M3FNUZ>
    %kf8 = arith.truncf %k : tensor<1x4096x64xf32> to tensor<1x4096x64xf8E4M3FNUZ>
    %vf8 = arith.truncf %v : tensor<1x4096x64xf32> to tensor<1x4096x64xf8E4M3FNUZ>

    %qk = arith.mulf %qscalef32, %kscalef32 : f32
    %qks = arith.mulf %qk, %scalef32 : f32

    %empty = tensor.empty() : tensor<1x4096x64xf32>
    %c0 = arith.constant 0.0 : f32
    %cqks = arith.constant 7.616043e-05 : f32
    // %cqks = arith.constant 1.250000e-01 : f32
    %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<1x4096x64xf32>)  -> tensor<1x4096x64xf32>
    %atten = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%qf8, %kf8, %vf8, %cqks : tensor<1x4096x64xf8E4M3FNUZ>, tensor<1x4096x64xf8E4M3FNUZ>, tensor<1x4096x64xf8E4M3FNUZ>, f32) outs(%fill : tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>

    // %atten_scale = call @scale(%atten, %vscalef32) : (tensor<1x4096x64xf32>, f32) -> tensor<1x4096x64xf32>

    return %atten : tensor<1x4096x64xf32>
}
