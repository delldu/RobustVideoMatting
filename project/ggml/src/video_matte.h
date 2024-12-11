#ifndef __VIDEO_MATTE_H__
#define __VIDEO_MATTE_H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"
// x = ggml_cont(ctx, x);
// ggml_set_name(x, "x");
// ggml_set_output(x);

int make_divisible(int v, int divisor);

struct Projection {
    // (conv): Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1))

    // network params
    struct Conv2d conv;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = 16;
        conv.out_channels = 4;
        conv.kernel_size = {1, 1};
        conv.stride = { 1, 1 };
        conv.padding = { 0, 0 };
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        return conv.forward(ctx, x);
    }
};

/*
 OutputBlock(
  (upsample): Upsample(scale_factor=2.0, mode='bilinear')
  (conv): Sequential(
    (0): Conv2d(35, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
) */

struct OutputBlock {
    int in_channels = 32;
    int out_channels = 16;
    
    // network params
    struct Conv2d conv_0;
    struct BatchNorm2d norm_1;

    struct Conv2d conv_3;
    struct BatchNorm2d norm_4;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv_0.in_channels = in_channels + 3;
        conv_0.out_channels = out_channels;
        conv_0.kernel_size = {3, 3};
        conv_0.stride = { 1, 1 };
        conv_0.padding = { 1, 1 };
        conv_0.has_bias = false;
        conv_0.create_weight_tensors(ctx);

        norm_1.num_features = out_channels;
        norm_1.create_weight_tensors(ctx);

        conv_3.in_channels = out_channels;
        conv_3.out_channels = out_channels;
        conv_3.kernel_size = {3, 3};
        conv_3.stride = { 1, 1 };
        conv_3.padding = { 1, 1 };
        conv_3.has_bias = false;
        conv_3.create_weight_tensors(ctx);

        norm_4.num_features = out_channels;
        norm_4.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.0.");
        conv_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.1.");
        norm_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.3.");
        conv_3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.4.");
        norm_4.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* s) {
        // x = self.upsample(x)
        // x = x[:, :, : s.size(2), : s.size(3)]
        // x = torch.cat([x, s], dim=1)
        // x = self.conv(x)
        // return x
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        // int C = (int)x->ne[2];
        // int B = (int)x->ne[3];
        x = ggml_interpolate(ctx, x, 0, 2*W);
        x = ggml_interpolate(ctx, x, 1, 2*H);

        x = ggml_concat(ctx, x, s, 2 /*dim on C*/);
        x = conv_0.forward(ctx, x);
        x = norm_1.forward(ctx, x);
        x = ggml_relu(ctx, x);
        x = conv_3.forward(ctx, x);
        x = norm_4.forward(ctx, x);
        x = ggml_relu(ctx, x);

        return x;
    }
};

struct ConvGRU {
    int channels = 16;

    // network params
    struct Conv2d ih_0;
    struct Conv2d hh_0;

    void create_weight_tensors(struct ggml_context* ctx) {
        ih_0.in_channels = channels * 2;
        ih_0.out_channels = channels * 2;
        ih_0.kernel_size = {3, 3};
        ih_0.stride = { 1, 1 };
        ih_0.padding = { 1, 1 };
        ih_0.create_weight_tensors(ctx);

        hh_0.in_channels = channels * 2;
        hh_0.out_channels = channels;
        hh_0.kernel_size = {3, 3};
        hh_0.stride = { 1, 1 };
        hh_0.padding = { 1, 1 };
        hh_0.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "ih.0.");
        ih_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "hh.0.");
        hh_0.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // h = torch.zeros_like(x)
        // r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        // c = self.hh(torch.cat([x, h], dim=1))
        // h = z * c
        // return h
        ggml_tensor_t *h = ggml_dup(ctx, x);
        h = ggml_constant(ctx, h, 0.0f);
        ggml_tensor_t *z = ggml_concat(ctx, x, h, 2/*dim on C */);
        // self.ih = nn.Sequential(nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=1), nn.Sigmoid())
        z = ih_0.forward(ctx, z);
        z = ggml_sigmoid(ctx, z);
        z = ggml_nn_slice(ctx, z, 2/*dim on C*/, channels, 2*channels, 1/*step*/);

        ggml_tensor_t *c = ggml_concat(ctx, x, h, 2/*dim on C */);
        // self.hh = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size, padding=1), nn.Tanh())
        c = hh_0.forward(ctx, c);
        c = ggml_tanh(ctx, c);
        h = ggml_mul(ctx, z, c);

        return h;
    }
};

/*
 UpsamplingBlock(
  (upsample): Upsample(scale_factor=2.0, mode='bilinear')
  (conv): Sequential(
    (0): Conv2d(59, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (gru): ConvGRU(
    (ih): Sequential(
      (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): Sigmoid()
    )
    (hh): Sequential(
      (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): Tanh()
    )
  )
) */

struct UpsamplingBlock {
    // network hparams
    int in_channels;
    int skip_channels;
    int out_channels = 32;

    // network params
    struct Conv2d conv_0;
    struct BatchNorm2d conv_1;
    struct ConvGRU gru;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv_0.in_channels = in_channels + skip_channels + 3;
        conv_0.out_channels = out_channels;
        conv_0.kernel_size = {3, 3};
        conv_0.stride = { 1, 1 };
        conv_0.padding = { 1, 1 };
        conv_0.has_bias = false;
        conv_0.create_weight_tensors(ctx);

        conv_1.num_features = out_channels;
        conv_1.create_weight_tensors(ctx);


        gru.channels = out_channels/2;
        gru.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.0.");
        conv_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.1.");
        conv_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "gru.");
        gru.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* f, ggml_tensor_t* s) {
        // x = self.upsample(x)
        // x = x[:, :, : s.size(2), : s.size(3)]
        // x = torch.cat([x, f, s], dim=1)
        // x = self.conv(x)

        // a, b = x.split(self.out_channels // 2, dim=1)
        // b = self.gru(b)
        // x = torch.cat([a, b], dim=1)
        // return x

        int W = (int)x->ne[0];
        int H = (int)x->ne[1];

        x = ggml_interpolate(ctx, x, 0, 2*W);
        x = ggml_interpolate(ctx, x, 1, 2*H);
        x = ggml_cat(ctx, 3, x, f, s, 2/*dim on C*/);
        // (conv): Sequential(
        //     (0): Conv2d(59, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        //     (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        //     (2): ReLU(inplace=True)
        // )        
        x = conv_0.forward(ctx, x);
        x = conv_1.forward(ctx, x);
        x = ggml_relu(ctx, x);
        ggml_tensor_t *a = ggml_nn_slice(ctx, x, 2/*dim on C*/, 0, out_channels, 1/*step*/);
        ggml_tensor_t *b = ggml_nn_slice(ctx, x, 2/*dim on C*/, out_channels, 2*out_channels, 1/*step*/);
        b = gru.forward(ctx, b);
        x = ggml_concat(ctx, a, b, 2/*dim on C*/);

    	return x;
    }
};

struct BottleneckBlock {
    int channels = 128;

    struct ConvGRU gru;

    void create_weight_tensors(struct ggml_context* ctx) {
        gru.channels = channels/2;
        gru.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "gru.");
        gru.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // a, b = x.split(self.channels // 2, dim=-3)
        // b = self.gru(b)
        // x = torch.cat([a, b], dim=-3)
        // return x
        ggml_tensor_t *a = ggml_nn_slice(ctx, x, 0/*dim on B*/, 0, channels, 1/*step*/);
        ggml_tensor_t *b = ggml_nn_slice(ctx, x, 0/*dim on B*/, channels, 2*channels, 1/*step*/);
        b = gru.forward(ctx, b);
        x = ggml_concat(ctx, a, b, 0/*dim on B*/);
        return x;
    }
};

/*
 AvgPool(
  (avgpool): AvgPool2d(kernel_size=2, stride=2, padding=0)
) */

struct AvgPool {
    // network params
    struct AvgPool2d avgpool;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        avgpool.kernel_size = 2;
        avgpool.stride = 2;
        avgpool.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        // char s[GGML_MAX_NAME];
        GGML_UNUSED(prefix);
    }

    // def forward(self, s0) -> List[torch.Tensor]:
    //     s1 = self.avgpool(s0)
    //     s2 = self.avgpool(s1)
    //     s3 = self.avgpool(s2)
    //     return s1, s2, s3
    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!
        std::vector<ggml_tensor_t *> xlist;
        ggml_tensor_t *s1, *s2, *s3;
        s1 = avgpool.forward(ctx, x);
        s2 = avgpool.forward(ctx, s1);
        s3 = avgpool.forward(ctx, s2);
        xlist.push_back(s1);
        xlist.push_back(s2);
        xlist.push_back(s3);
    	return xlist;
    }
};


struct RecurrentDecoder {
    // network hparams

    // network params
    struct AvgPool avgpool;
    struct BottleneckBlock decode4;
    struct UpsamplingBlock decode3;
    struct UpsamplingBlock decode2;
    struct UpsamplingBlock decode1;
    struct OutputBlock decode0;

    void create_weight_tensors(struct ggml_context* ctx) {
        avgpool.create_weight_tensors(ctx);

        // self.decode4 = BottleneckBlock(128)
        decode4.channels = 128;
        decode4.create_weight_tensors(ctx);

        // self.decode3 = UpsamplingBlock(128, 40, 3, 80)
        decode3.in_channels = 128;
        decode3.skip_channels = 40;
        decode3.out_channels = 80;
        decode3.create_weight_tensors(ctx);

        // self.decode2 = UpsamplingBlock(80, 24, 3, 40)
        decode2.in_channels = 80;
        decode2.skip_channels = 24;
        decode2.out_channels = 40;
        decode2.create_weight_tensors(ctx);

        // self.decode1 = UpsamplingBlock(40, 16, 3, 32)
        decode1.in_channels = 40;
        decode1.skip_channels = 16;
        decode1.out_channels = 32;
        decode1.create_weight_tensors(ctx);

        // self.decode0 = OutputBlock(32, 3, 16)
        decode0.in_channels = 32;
        decode0.out_channels = 16;
        decode0.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        // snprintf(s, sizeof(s), "%s%s", prefix, "avgpool.");
        // avgpool.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "decode4.");
        decode4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "decode3.");
        decode3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "decode2.");
        decode2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "decode1.");
        decode1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "decode0.");
        decode0.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* f1, 
        ggml_tensor_t* f2, ggml_tensor_t* f3, ggml_tensor_t* f4) {
        // s1, s2, s3 = self.avgpool(s0)
        // x4 = self.decode4(f4)
        // x3 = self.decode3(x4, f3, s3)
        // x2 = self.decode2(x3, f2, s2)
        // x1 = self.decode1(x2, f1, s1)
        // x0 = self.decode0(x1, s0)
        // return x0 #, r1, r2, r3, r4
        std::vector<ggml_tensor_t *> xlist;
        ggml_tensor_t *s1, *s2, *s3;
        ggml_tensor_t *x0, *x1, *x2, *x3, *x4;

        xlist = avgpool.forward(ctx, x);
        s1 = xlist[0]; s2 = xlist[1]; s3 = xlist[1];

        x4 = decode4.forward(ctx, f4);
        x3 = decode3.forward(ctx, x4, f3, s3);
        x2 = decode2.forward(ctx, x3, f2, s2);
        x1 = decode1.forward(ctx, x2, f1, s1);
        x0 = decode0.forward(ctx, x1, x);

        return x0;
    }
};

/*
 LRASPP(
  (aspp1): Sequential(
    (0): Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (aspp2): Sequential(
    (0): AdaptiveAvgPool2d(output_size=1)
    (1): Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (2): Sigmoid()
  )
) */

struct LRASPP {
    int in_channels = 960;
    int out_channels = 128;

    // network params
    struct Conv2d aspp1_0;
    struct BatchNorm2d aspp1_1;

    struct AdaptiveAvgPool2d aspp2_0;
    struct Conv2d aspp2_1;

    // def __init__(self, in_channels, out_channels):
    //     super().__init__()
    //     # 960, 128
    //     self.aspp1 = nn.Sequential(
    //         nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True)
    //     )
    //     self.aspp2 = nn.Sequential(
    //         nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.Sigmoid()
    //     )

    void create_weight_tensors(struct ggml_context* ctx) {
        aspp1_0.in_channels = in_channels;
        aspp1_0.out_channels = out_channels;
        aspp1_0.kernel_size = {1, 1};
        aspp1_0.stride = { 1, 1 };
        aspp1_0.padding = { 0, 0 };
        aspp1_0.has_bias = false;
        aspp1_0.create_weight_tensors(ctx);

        aspp1_1.num_features = out_channels;
        aspp1_1.create_weight_tensors(ctx);

        aspp2_0.output_height = 1;
        aspp2_0.output_width = 1;
        aspp2_0.create_weight_tensors(ctx);

        aspp2_1.in_channels = in_channels;
        aspp2_1.out_channels = out_channels;
        aspp2_1.kernel_size = {1, 1};
        aspp2_1.stride = { 1, 1 };
        aspp2_1.padding = { 0, 0 };
        aspp2_1.has_bias = false;
        aspp2_1.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "aspp1.0.");
        aspp1_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "aspp1.1.");
        aspp1_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "aspp2.0.");
        aspp2_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "aspp2.1.");
        aspp2_1.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // return self.aspp1(x) * self.aspp2(x)

        // self.aspp1 = nn.Sequential(
        //     nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        // )        
        ggml_tensor_t *x1 = aspp1_0.forward(ctx, x);
        x1 = aspp1_1.forward(ctx, x1);
        x1 = ggml_relu(ctx, x1);

        // self.aspp2 = nn.Sequential(
        //     nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.Sigmoid()
        // )
        ggml_tensor_t *x2 = aspp2_0.forward(ctx, x);
        x2 = aspp2_1.forward(ctx, x2);
        x2 = ggml_sigmoid(ctx, x2);

        x = ggml_mul(ctx, x1, x2);
    	return x;
    }
};

/*
 SqueezeExcitation(
  (avgpool): AdaptiveAvgPool2d(output_size=1)
  (fc1): Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))
  (fc2): Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))
  (activation): ReLU()
  (scale_activation): Hardsigmoid()
) */

struct SqueezeExcitation {
    int input_channels;
    
    struct AdaptiveAvgPool2d avgpool;
    struct Conv2d fc1;
    struct Conv2d fc2;

    // self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
    // self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)

    void create_weight_tensors(struct ggml_context* ctx) {
        // assert _make_divisible(input_channels//4, 8) == squeeze_channels
        int squeeze_channels = make_divisible(input_channels/4, 8);

        avgpool.output_height = 1;
        avgpool.output_width = 1;
        avgpool.create_weight_tensors(ctx);

        fc1.in_channels = input_channels;
        fc1.out_channels = squeeze_channels;
        fc1.kernel_size = {1, 1};
        fc1.stride = { 1, 1 };
        fc1.padding = { 0, 0 };
        fc1.create_weight_tensors(ctx);

        fc2.in_channels = squeeze_channels;
        fc2.out_channels = input_channels;
        fc2.kernel_size = {1, 1};
        fc2.stride = { 1, 1 };
        fc2.padding = { 0, 0 };
        fc2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "fc1.");
        fc1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "fc2.");
        fc2.setup_weight_names(s);
    }

    // def _scale(self, input: Tensor) -> Tensor:
    //     scale = self.avgpool(input)
    //     scale = self.fc1(scale)
    //     scale = self.activation(scale)
    //     scale = self.fc2(scale)
    //     return self.scale_activation(scale)

    // def forward(self, input: Tensor) -> Tensor:
    //     scale = self._scale(input)
    //     return scale * input
    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *scale = avgpool.forward(ctx, x);
        scale = fc1.forward(ctx, scale);
        scale = ggml_relu(ctx, scale);
        scale = fc2.forward(ctx, scale);
        scale = ggml_hardsigmoid(ctx, scale);

        x = ggml_mul(ctx, scale, x);

        return x;
    }
};

/*
 Conv2dNormActivation(
  (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (1): BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (2): Hardswish()
) */

struct Conv2dNormActivation {
    int in_channels = 160;
    int out_channels = 960;
    int kernel_size = 1;
    int stride = 1;
    // int padding = 0;
    int dilation = 1;
    int groups = 1;
    int activation_layer = 0; // 1 -- ReLU, 2 -- Hardswish

    // network params
    struct Conv2d conv;
    struct BatchNorm2d norm;

    void create_weight_tensors(struct ggml_context* ctx) {
        // update padding ...
        int padding = (kernel_size - 1) / 2 * dilation;

        conv.in_channels = in_channels;
        conv.out_channels = out_channels;
        conv.kernel_size = {kernel_size, kernel_size};
        conv.stride = { stride, stride };
        conv.padding = { padding, padding };
        conv.dilation = { dilation, dilation};
        conv.has_bias = false;
        if (groups > 1) {
            conv.is_depthwise = true;
        }
        conv.create_weight_tensors(ctx);

        norm.num_features = out_channels;
        norm.eps = 0.001;
        norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "1.");
        norm.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = norm.forward(ctx, x);
        if (activation_layer > 0) {
            if (activation_layer == 2) {
                x = ggml_hardswish(ctx, x);
            } else {
                x = ggml_relu(ctx, x);
            }
        }
        return x;
    }
};


struct InvertedResidual {
    // network hparams
    int input_channels;
    int kernel;
    int expanded_channels;
    int out_channels;
    int use_se = 0;
    int use_hs = 0;
    int stride = 1;
    int dilation = 1;

    // internal control variables
    int use_res_connect = 0;
    int has_expand_block = 0;

    // network params
    struct Conv2dNormActivation expand;
    struct Conv2dNormActivation depthwise;
    struct SqueezeExcitation selayer;
    struct Conv2dNormActivation project;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        if (stride == 1 && input_channels == out_channels) {
            use_res_connect = 1;
        }
        int activation_layer = (use_hs)? 2 : 1;  // nn.Hardswish if cnf.use_hs else nn.ReLU
        has_expand_block = (input_channels != expanded_channels);

        // # expand
        // if cnf.expanded_channels != cnf.input_channels:
        //     layers.append(
        //         Conv2dNormActivation(
        //             cnf.input_channels,
        //             cnf.expanded_channels,
        //             kernel_size=1,
        //             norm_layer=norm_layer,
        //             activation_layer=activation_layer,
        //         )
        //     )
        if (has_expand_block) {
            expand.in_channels = input_channels;
            expand.out_channels = expanded_channels;
            expand.kernel_size = 1;
            expand.activation_layer = activation_layer; // 1 -- ReLU, 2 -- Hardswish
            expand.create_weight_tensors(ctx);
        }

        // # depthwise
        // stride = 1 if cnf.dilation > 1 else cnf.stride
        // layers.append(
        //     Conv2dNormActivation(
        //         cnf.expanded_channels,
        //         cnf.expanded_channels,
        //         kernel_size=cnf.kernel,
        //         stride=stride,
        //         dilation=cnf.dilation,
        //         groups=cnf.expanded_channels,
        //         norm_layer=norm_layer,
        //         activation_layer=activation_layer,
        //     )
        // )
        if (dilation > 1) {
            stride = 1;  
        }
        depthwise.in_channels = expanded_channels;
        depthwise.out_channels = expanded_channels;
        depthwise.kernel_size = kernel;
        depthwise.stride = stride;
        // int padding = 0;
        depthwise.dilation = dilation;
        depthwise.groups = expanded_channels;
        depthwise.activation_layer = activation_layer;
        depthwise.create_weight_tensors(ctx);

        // if cnf.use_se:
        //     squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
        //     layers.append(se_layer(cnf.expanded_channels, squeeze_channels))
        if (use_se) {
            selayer.input_channels = expanded_channels;
            selayer.create_weight_tensors(ctx);
        }

        // # project
        // layers.append(
        //     Conv2dNormActivation(
        //         cnf.expanded_channels, cnf.out_channels, 
        //         kernel_size=1, 
        //         norm_layer=norm_layer, 
        //         activation_layer=None
        //     )
        // )
        project.in_channels = expanded_channels;
        project.out_channels = out_channels;
        project.kernel_size = 1;
        // int padding = 0;
        project.activation_layer = 0;
        project.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        int block_no = 0;

        if (has_expand_block) {
            snprintf(s, sizeof(s), "%sblock.%d.", prefix, block_no);
            expand.setup_weight_names(s);
            block_no++;
        }

        snprintf(s, sizeof(s), "%sblock.%d.", prefix, block_no);
        depthwise.setup_weight_names(s);
        block_no++;

        if (use_se) {
            snprintf(s, sizeof(s), "%sblock.%d.", prefix, block_no);
            selayer.setup_weight_names(s);
            block_no++;
        }

        snprintf(s, sizeof(s), "%sblock.%d.", prefix, block_no);
        project.setup_weight_names(s);
        // block_no++;
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // result = self.block(input)
        // if self.use_res_connect:
        //     result += input
        // return result
        ggml_tensor_t *y = x;
        if (has_expand_block) {
            y = expand.forward(ctx, x);
        }
        y = depthwise.forward(ctx, y);
        if (use_se) {
            y = selayer.forward(ctx, y);
        }
        y = project.forward(ctx, y);
        if (use_res_connect) {
            y = ggml_add(ctx, y, x);
        }
        return y;
    }
};


struct MobileNetV3LargeEncoder {
    // network params
    struct Normalize normal;
    struct Conv2dNormActivation features_0;
    struct InvertedResidual features_1_15[15];
    struct Conv2dNormActivation features_16;

    void create_weight_tensors(struct ggml_context* ctx) {
        normal.create_weight_tensors(ctx);

        // (features): Sequential(
        //   (0): Conv2dNormActivation(
        //     (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        //     (1): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        //     (2): Hardswish()
        //   )
        features_0.in_channels = 3;
        features_0.out_channels = 16;
        features_0.kernel_size = 3;
        features_0.stride = 2;
        // int padding = 0;
        features_0.dilation = 1;
        features_0.groups = 1;
        features_0.activation_layer = 2; // 1 -- ReLU, 2 -- Hardswish
        features_0.create_weight_tensors(ctx);


        // # input_channels, kernel, expanded_channels, out_channels,
        // # use_se: bool, # activation: str,
        // # stride, dilation, width_mult
        // InvertedResidualConfig(16, 3, 16, 16, False, "RE", 1, 1, 1),
        features_1_15[0].input_channels = 16;
        features_1_15[0].kernel = 3;
        features_1_15[0].expanded_channels = 16;
        features_1_15[0].out_channels = 16;
        features_1_15[0].use_se = 0;
        features_1_15[0].use_hs = 0;
        features_1_15[0].stride = 1;
        features_1_15[0].dilation = 1;
        features_1_15[0].create_weight_tensors(ctx);

        // InvertedResidualConfig(16, 3, 64, 24, False, "RE", 2, 1, 1),  # C1
        features_1_15[1].input_channels = 16;
        features_1_15[1].kernel = 3;
        features_1_15[1].expanded_channels = 64;
        features_1_15[1].out_channels = 24;
        features_1_15[1].use_se = 0;
        features_1_15[1].use_hs = 0;
        features_1_15[1].stride = 2;
        features_1_15[1].dilation = 1;
        features_1_15[1].create_weight_tensors(ctx);

        // InvertedResidualConfig(24, 3, 72, 24, False, "RE", 1, 1, 1),
        features_1_15[2].input_channels = 24;
        features_1_15[2].kernel = 3;
        features_1_15[2].expanded_channels = 72;
        features_1_15[2].out_channels = 24;
        features_1_15[2].use_se = 0;
        features_1_15[2].use_hs = 0;
        features_1_15[2].stride = 1;
        features_1_15[2].dilation = 1;
        features_1_15[2].create_weight_tensors(ctx);

        // InvertedResidualConfig(24, 5, 72, 40, True, "RE", 2, 1, 1),  # C2
        features_1_15[3].input_channels = 24;
        features_1_15[3].kernel = 5;
        features_1_15[3].expanded_channels = 72;
        features_1_15[3].out_channels = 40;
        features_1_15[3].use_se = 1;
        features_1_15[3].use_hs = 0;
        features_1_15[3].stride = 2;
        features_1_15[3].dilation = 1;
        features_1_15[3].create_weight_tensors(ctx);

        // InvertedResidualConfig(40, 5, 120, 40, True, "RE", 1, 1, 1),
        features_1_15[4].input_channels = 40;
        features_1_15[4].kernel = 5;
        features_1_15[4].expanded_channels = 120;
        features_1_15[4].out_channels = 40;
        features_1_15[4].use_se = 1;
        features_1_15[4].use_hs = 0;
        features_1_15[4].stride = 1;
        features_1_15[4].dilation = 1;
        features_1_15[4].create_weight_tensors(ctx);

        // InvertedResidualConfig(40, 5, 120, 40, True, "RE", 1, 1, 1),
        features_1_15[5].input_channels = 40;
        features_1_15[5].kernel = 5;
        features_1_15[5].expanded_channels = 120;
        features_1_15[5].out_channels = 40;
        features_1_15[5].use_se = 1;
        features_1_15[5].use_hs = 0;
        features_1_15[5].stride = 1;
        features_1_15[5].dilation = 1;
        features_1_15[5].create_weight_tensors(ctx);

        // InvertedResidualConfig(40, 3, 240, 80, False, "HS", 2, 1, 1),  # C3
        features_1_15[6].input_channels = 40;
        features_1_15[6].kernel = 3;
        features_1_15[6].expanded_channels = 240;
        features_1_15[6].out_channels = 80;
        features_1_15[6].use_se = 0;
        features_1_15[6].use_hs = 1;
        features_1_15[6].stride = 2;
        features_1_15[6].dilation = 1;
        features_1_15[6].create_weight_tensors(ctx);

        // InvertedResidualConfig(80, 3, 200, 80, False, "HS", 1, 1, 1),
        features_1_15[7].input_channels = 80;
        features_1_15[7].kernel = 3;
        features_1_15[7].expanded_channels = 200;
        features_1_15[7].out_channels = 80;
        features_1_15[7].use_se = 0;
        features_1_15[7].use_hs = 1;
        features_1_15[7].stride = 1;
        features_1_15[7].dilation = 1;
        features_1_15[7].create_weight_tensors(ctx);

        // InvertedResidualConfig(80, 3, 184, 80, False, "HS", 1, 1, 1),
        features_1_15[8].input_channels = 80;
        features_1_15[8].kernel = 3;
        features_1_15[8].expanded_channels = 184;
        features_1_15[8].out_channels = 80;
        features_1_15[8].use_se = 0;
        features_1_15[8].use_hs = 1;
        features_1_15[8].stride = 1;
        features_1_15[8].dilation = 1;
        features_1_15[8].create_weight_tensors(ctx);

        // InvertedResidualConfig(80, 3, 184, 80, False, "HS", 1, 1, 1),
        features_1_15[9].input_channels = 80;
        features_1_15[9].kernel = 3;
        features_1_15[9].expanded_channels = 184;
        features_1_15[9].out_channels = 80;
        features_1_15[9].use_se = 0;
        features_1_15[9].use_hs = 1;
        features_1_15[9].stride = 1;
        features_1_15[9].dilation = 1;
        features_1_15[9].create_weight_tensors(ctx);

        // InvertedResidualConfig(80, 3, 480, 112, True, "HS", 1, 1, 1),
        features_1_15[10].input_channels = 80;
        features_1_15[10].kernel = 3;
        features_1_15[10].expanded_channels = 480;
        features_1_15[10].out_channels = 112;
        features_1_15[10].use_se = 1;
        features_1_15[10].use_hs = 1;
        features_1_15[10].stride = 1;
        features_1_15[10].dilation = 1;
        features_1_15[10].create_weight_tensors(ctx);

        // InvertedResidualConfig(112, 3, 672, 112, True, "HS", 1, 1, 1),
        features_1_15[11].input_channels = 112;
        features_1_15[11].kernel = 3;
        features_1_15[11].expanded_channels = 672;
        features_1_15[11].out_channels = 112;
        features_1_15[11].use_se = 1;
        features_1_15[11].use_hs = 1;
        features_1_15[11].stride = 1;
        features_1_15[11].dilation = 1;
        features_1_15[11].create_weight_tensors(ctx);

        // InvertedResidualConfig(112, 5, 672, 160, True, "HS", 2, 2, 1),  # C4
        features_1_15[12].input_channels = 112;
        features_1_15[12].kernel = 5;
        features_1_15[12].expanded_channels = 672;
        features_1_15[12].out_channels = 160;
        features_1_15[12].use_se = 1;
        features_1_15[12].use_hs = 1;
        features_1_15[12].stride = 2;
        features_1_15[12].dilation = 2;
        features_1_15[12].create_weight_tensors(ctx);

        // InvertedResidualConfig(160, 5, 960, 160, True, "HS", 1, 2, 1),
        features_1_15[13].input_channels = 160;
        features_1_15[13].kernel = 5;
        features_1_15[13].expanded_channels = 960;
        features_1_15[13].out_channels = 160;
        features_1_15[13].use_se = 1;
        features_1_15[13].use_hs = 1;
        features_1_15[13].stride = 1;
        features_1_15[13].dilation = 2;
        features_1_15[13].create_weight_tensors(ctx);

        // InvertedResidualConfig(160, 5, 960, 160, True, "HS", 1, 2, 1),
        features_1_15[14].input_channels = 160;
        features_1_15[14].kernel = 5;
        features_1_15[14].expanded_channels = 960;
        features_1_15[14].out_channels = 160;
        features_1_15[14].use_se = 1;
        features_1_15[14].use_hs = 1;
        features_1_15[14].stride = 1;
        features_1_15[14].dilation = 2;
        features_1_15[14].create_weight_tensors(ctx);

        // for (int i = 0; i < 15; i++) {
        //     features_1_15[i].create_weight_tensors(ctx);
        // }

        // (16): Conv2dNormActivation(
        //   (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        //   (1): BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        //   (2): Hardswish()
        // )
        features_16.in_channels = 160;
        features_16.out_channels = 960;
        features_16.kernel_size = 1;
        features_16.stride = 1;
        // int padding = 0;
        features_16.dilation = 1;
        features_16.groups = 1;
        features_16.activation_layer = 2; // 1 -- ReLU, 2 -- Hardswish
        features_16.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "features.0.");
        features_0.setup_weight_names(s);

        for (int i = 0; i < 15; i++) {
            snprintf(s, sizeof(s), "%sfeatures.%d.", prefix, i + 1);
            features_1_15[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "features.16.");
        features_16.setup_weight_names(s);
    }


    std::vector<ggml_tensor_t*> forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        // x = self.features[0](x)
        // x = self.features[1](x)
        // f1 = x
        // x = self.features[2](x)
        // x = self.features[3](x)
        // f2 = x
        // x = self.features[4](x)
        // x = self.features[5](x)
        // x = self.features[6](x)
        // f3 = x
        // x = self.features[7](x)
        // x = self.features[8](x)
        // x = self.features[9](x)
        // x = self.features[10](x)
        // x = self.features[11](x)
        // x = self.features[12](x)
        // x = self.features[13](x)
        // x = self.features[14](x)
        // x = self.features[15](x)
        // x = self.features[16](x)
        // f4 = x

        // return [f1, f2, f3, f4]

        std::vector<ggml_tensor_t*> xlist;
        x = normal.forward(ctx, x);

        x = features_0.forward(ctx, x);
        x = features_1_15[0].forward(ctx, x);
        xlist.push_back(x); // f1
        x = features_1_15[1].forward(ctx, x);
        x = features_1_15[2].forward(ctx, x);
        xlist.push_back(x); // f2
        x = features_1_15[3].forward(ctx, x);
        x = features_1_15[4].forward(ctx, x);
        x = features_1_15[5].forward(ctx, x);
        xlist.push_back(x); // f3
        x = features_1_15[6].forward(ctx, x);
        x = features_1_15[7].forward(ctx, x);
        x = features_1_15[8].forward(ctx, x);
        x = features_1_15[9].forward(ctx, x);
        x = features_1_15[10].forward(ctx, x);
        x = features_1_15[11].forward(ctx, x);
        x = features_1_15[12].forward(ctx, x);
        x = features_1_15[13].forward(ctx, x);
        x = features_1_15[14].forward(ctx, x);
        x = features_16.forward(ctx, x);
        xlist.push_back(x); // f4

        return xlist;
    }
};


struct MattingNetwork : GGMLNetwork {
    // network hparams
    int MAX_H = 2048;
    int MAX_W = 4048;
    int MAX_TIMES = 4;

    // network params
    struct MobileNetV3LargeEncoder backbone;
    struct LRASPP aspp;
    struct RecurrentDecoder decoder;
    struct Projection project_mat;

    void create_weight_tensors(struct ggml_context* ctx) {
        backbone.create_weight_tensors(ctx);
        aspp.create_weight_tensors(ctx);
        decoder.create_weight_tensors(ctx);
        project_mat.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "backbone.");
        backbone.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "aspp.");
        aspp.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "decoder.");
        decoder.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "project_mat.");
        project_mat.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_UNUSED(argc);
        ggml_tensor_t *x = argv[0];

        // f1, f2, f3, f4 = self.backbone(src)
        // f4 = self.aspp(f4)
        // hid = self.decoder(src, f1, f2, f3, f4)
        // src_residual, mask = self.project_mat(hid).split([3, 1], dim=-3)
        // mask = mask.clamp(0.0, 1.0)
        // output = torch.cat((src, mask), dim=1)
        // return output

        ggml_tensor_t *f1, *f2, *f3, *f4;
        std::vector<ggml_tensor_t *>xlist = backbone.forward(ctx, x);
        f1 = xlist[0]; f2 = xlist[1]; f3 = xlist[2]; f4 = xlist[3];
        f4 = aspp.forward(ctx, f4);
        ggml_tensor_t *hid = decoder.forward(ctx, x, f1, f2, f3, f4);
        ggml_tensor_t *out = project_mat.forward(ctx, hid);
        ggml_tensor_t *mask = ggml_nn_slice(ctx, out, 2 /*dim*/, 0, 1, 1/*step*/);
        out = ggml_concat(ctx, x, mask, 2/*dim*/);

        return out;
    }
};


struct VideoMatteNetwork {
    MattingNetwork net;
    GGMLModel model;

    int init(int device) {
        // -----------------------------------------------------------------------------------------
        net.set_device(device);
        net.start_engine();
        net.dump();

        check_point(model.preload("models/video_matte_f32.gguf") == RET_OK);

        return RET_OK;
    }

    int load() {
        return net.load_weight(&model, "");
    }

    TENSOR *forward(TENSOR *input_tensor) {
        TENSOR *argv[1];
        argv[0] = input_tensor ;

        load();
        return net.engine_forward(ARRAY_SIZE(argv), argv);
    }

    void exit() {
        model.clear();
        net.stop_engine();
    }
};

#endif // __VIDEO_MATTE_H__
