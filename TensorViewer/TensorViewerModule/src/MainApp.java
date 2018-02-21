/**
 * Created by sam on 4/23/17.
 */

import processing.core.PApplet;
import processing.event.MouseEvent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;


public class MainApp extends PApplet {
    private static String fname  = "/dsde/working/sam/palantir_cnn/Analysis/vqsr_cnn/generated_1d/excite_conv1d_2/filter_22.hd5";

    //private static String data_path  = "/dsde/data/deep/vqsr/tensors/g94982_ref_read_tf/valid";
    //private static String data_path  = "/Users/sam/vqsr_data/tensors/balancer_2d_tensors/train";
    //private static String data_path  = "/dsde/data/deep/vqsr/tensors/exome_scratch/train";
    //private static String data_path  = "/dsde/data/deep/vqsr/tensors/g94982_paired_read/train" ;
    //private static String data_path  =  "/dsde/data/deep/vqsr/tensors/g94794_chm_wgs1_flag_stretch/train";
    //private static String data_path  =  "/dsde/data/deep/vqsr/tensors/mix_big_ref_read_anno/train";
    private static String data_path  =  "/dsde/working/sam/dsde-deep-learning/api_tutorials/weights";

    //private static String data_path  = "/dsde/data/deep/vqsr/tensors/g94982_calling_tensors3";
    //private static String data_path  = "/dsde/data/deep/vqsr/tensors/g94982_calling_tensors_indel_only/";
    //private static String data_path  = "/dsde/data/deep/vqsr/tensors/g94982_calling_tensors_variant_sort/";
    boolean calling_tensors = false;

    private static String weights_path = "/dsde/working/sam/palantir_cnn/Analysis/vqsr_cnn/weights/m__base_quality_mode_phot__channels_last_False__id_g94982_mq_train__window_size_128__read_limit_128__random_seed_12878__tensor_map_2d_mapping_quality__mode_ref_read_anno.hd5";

    //private static String dna_weights_path = "/dsde/working/sam/palantir_cnn/Analysis/vqsr_cnn/weights/m__base_quality_mode_phot__channels_last_False__id_dna_only_small_kernels__window_size_71__read_limit_128__random_seed_12878__tensor_map_1d_dna__mode_reference.hd5";
    //private static String dna_weights_path = "/dsde/working/sam/palantir_cnn/Analysis/vqsr_cnn/weights/m__base_quality_mode_phot__channels_last_False__id_dna_only_long_kernels__window_size_71__read_limit_128__random_seed_12878__tensor_map_1d_dna__mode_reference.hd5";
    //private static String dna_weights_path = "/dsde/working/sam/palantir_cnn/Analysis/vqsr_cnn/weights/m__base_quality_mode_phot__channels_last_False__id_hg38_1d__window_size_128__read_limit_128__random_seed_12878__tensor_map_1d_dna__mode_reference.hd5";
    private static String dna_weights_path = "/dsde/working/sam/palantir_cnn/Analysis/vqsr_cnn/weights/1d_ref_anno_skip2.hd5";

    int cur_label = 0;
    int cur_tensor = 0;
    float cur_scale = 100;
    float max_scale = 1080;
    float min_scale = 10;

    List<List<Path>> tensor_files;
    float[] annotations;
    int[] tensor_shape = {15, 128, 128};
    //int[] tensor_shape = {128, 128, 15};

    int channel_idx = 0;
    int seq_idx = 1;
    int cur_channel = 0;

    int[] channel_colors = {color(0, 255, 0), color(0, 0, 255), color(155, 105, 0), color(255, 0, 0), color(155, 0, 155),
            color(0, 155, 0), color(120, 120, 255), color(105, 55, 0),  color(205, 0, 55), color(255, 0, 255),
            color(0, 0, 128), color(128, 0, 0), color(128, 128, 0), color(0, 128, 128), color(128, 0, 128),
            color(0, 0, 68), color(68, 0, 0), color(68, 68, 0)};

    String[] channel_names = {"Read A", "Read C", "Read G", "Read T", "DELETION",
            "Reference A", "Reference C", "Reference G", "Reference T", "INSERTION",
            "Read Reverse Strand", "Mate Reverse Strand", "First in Pair", "Second in Pair",
            "Fails QC",  "PCR or Optical Duplicate", "Mapping Quality"};

    String[] calling_labels = { "Reference", "Heterozygous SNP", "Homozygous SNP", "Heterozygous Deletion", "Homozygous Deletion", "Heterozygous Insertion", "Homozygous Insertion" };

    int[] reference_shape = {71,4};

    String[] dna = {"A", "C", "G", "T"};
    int cur_kernel = 0;
    int cur_layer_idx =  0;
    String[] layer_names = {"conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4"};
    int samples = 30;
    String[] annotation_names = {"MQ", "DP", "SOR", "FS", "QD", "MQRankSum", "ReadPosRankSum"};
    boolean load_annotations = false;
    int cur_mode = 2;
    boolean highlight_channel = false;
    boolean max_logo_only = true;
    String[] modes = {"tensor", "weights", "logos", "reference"};
    TensorHolder th;

    float xmag, ymag = 0;
    float newXmag, newYmag = 0;

    public static void main(String[] args){
        PApplet.main("MainApp", args);
    }

    public void settings(){
        size(1600, 1200,  "processing.opengl.PGraphics3D");
        tensor_files = getTensorFiles(data_path);
        th = new TensorHolder(annotation_names);
        th.load_tensor_3d(tensor_files.get(cur_label).get(cur_tensor), "read_tensor", tensor_shape);
        th.load_dna_weights(dna_weights_path, "conv1d_1");
        th.load_reference_tensor(Paths.get(fname), "reference", reference_shape);
        th.load_convolution_kernels(weights_path, layer_names[cur_layer_idx]);
        if (calling_tensors){
            th.load_label_tensor(tensor_files.get(cur_label).get(cur_tensor), "site_labels", tensor_shape[1]);
        }
    }

    private List<List<Path>> getTensorFiles(String data_path){
        List<List<Path>> tensor_files = new ArrayList<List<Path>>();
        try {
            List<Path> label_dirs =  Files.list(Paths.get(data_path))
                    .filter(Files::isDirectory)
                    .collect(Collectors.toList());
            for (final Path label : label_dirs){
                List<Path> label_tensors = new ArrayList<Path>();
                label_tensors.addAll(getFileList(label, samples));
                System.out.println("GOT LABEL TENSORS:\n"+label_tensors.toString());
                tensor_files.add(label_tensors);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return tensor_files;
    }

    private List<Path> getFileList(Path tensor_path, int max_files){
        try {
            return Files.walk(Paths.get(tensor_path.toString()))
                    .limit(max_files)
                    .filter(Files::isRegularFile)
                    .collect(Collectors.toList());
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

    }


    public void setup(){
        noStroke();
        colorMode(RGB, 255);
        ortho();
    }

    public void draw() {
        background(255);

        fill(20);
        if (modes[cur_mode].equals("tensor")) {
            textSize(22);

            text(tensor_files.get(cur_label).get(cur_tensor).getFileName().toString(), 220, 24);
            int text_y = 145;
            int text_line_height = 48;
            textSize(26);

            for (int i = 0; i < tensor_shape[channel_idx]; i++) {

                fill(channel_colors[i]);
                if (highlight_channel && cur_channel != i) fill(channel_colors[i], 45);
                if (highlight_channel && cur_channel == i) textSize(32);

                text(channel_names[i], 30, text_y);
                text_y += text_line_height;
            }
            if (load_annotations) {
                fill(20);
                for (int i = 0; i < annotation_names.length; i++) {
                    text_y += text_line_height;
                    text(annotation_names[i] + " : " + Float.toString(annotations[i]), 30, text_y);
                }
            }

            if (calling_tensors){
                text_y = 100;
                for (int i = 0; i < calling_labels.length; i++) {
                    fill(channel_colors[i]);
                    text_y += text_line_height;
                    text(calling_labels[i], width-450, text_y);
                }
            }
        } else if (modes[cur_mode].equals("weights")) {
            textSize(36);
            text("Weights from layer: " + layer_names[cur_layer_idx], 50, 40);
        } else if (modes[cur_mode].equals("logos")) {
            textSize(24);
            text("Show DNA Logos", 5, 30);
            showDNALogos(cur_scale);
        } else if (modes[cur_mode].equals("reference")) {
            textSize(24);
            text("Show Reference", 5, 30);
            showDNAReference(th.reference_tensor, 5, 200, cur_scale);
        }



        pushMatrix();
        translate(width / 2, height / 2, -10);

        newXmag = (float) (mouseX / (float) (width) * 6.28);
        newYmag = (float) (mouseY / (float) (height) * 6.28);

        float diff = xmag - newXmag;
        if (abs(diff) > 0.01) {
            xmag -= diff / 4.0;
        }

        diff = ymag - newYmag;
        if (abs(diff) > 0.01) {
            ymag -= diff / 4.0;
        }

        rotateX(-ymag);
        rotateY(-xmag);
        scale(cur_scale);

        if (modes[cur_mode].equals("weights")) {
            showKernels();
        } else if (modes[cur_mode].equals("tensor")) {
            showTensor();
        }
        popMatrix();
    }

    public void keyPressed(){
        if (key == 'l'){
            cur_label = ++cur_label % tensor_files.size();
            cur_tensor = 0;
        } else if (key =='t'){
            cur_tensor = ++cur_tensor % tensor_files.get(0).size();
        } else if (key =='d'){
            cur_kernel = (21+cur_kernel) % th.dna_weights[0][0].length;
        } else if (key =='c'){
            cur_channel = ++cur_channel % channel_names.length;
        } else if (key =='h'){
            highlight_channel = !highlight_channel;
            max_logo_only = !max_logo_only;
        } else if (key =='m'){
            cur_mode = ++cur_mode% modes.length;
        } else if(key == 'k'){
            cur_layer_idx = ++cur_layer_idx % layer_names.length;
            th.load_convolution_kernels(weights_path, layer_names[cur_layer_idx]);
        }
        if (modes[cur_mode].equals("reference") && (key == 'l' || key == 't')) {
            System.out.println("Load label #:"+cur_label+" tensor #:"+cur_tensor+ " from path:\n"+tensor_files.get(cur_label).get(cur_tensor).toString());
            th.load_reference_tensor(tensor_files.get(cur_label).get(cur_tensor), "reference", reference_shape);
        } else if (key == 'l' || key == 't'){
            System.out.println("Load label #:"+cur_label+" tensor #:"+cur_tensor+ " from path:\n"+tensor_files.get(cur_label).get(cur_tensor).toString());
            th.load_tensor_3d(tensor_files.get(cur_label).get(cur_tensor), "read_tensor", tensor_shape);
            if (calling_tensors){
                th.load_label_tensor(tensor_files.get(cur_label).get(cur_tensor), "site_labels", tensor_shape[1]);
            }
        }

    }

    public void mouseWheel(MouseEvent event) {
        float e = event.getCount();
        cur_scale += e;
        cur_scale = constrain(cur_scale, min_scale, max_scale);
    }

    public void showKernels(){
        float eps = 0.04f;
        float sizer = 2.0f;
        float weight_scalar = 5.0f;

        float x_inc = sizer / (float)th.kernel_shape[0];
        float y_inc = sizer / (float)th.kernel_shape[3];
        float z_inc = sizer / (float) th.kernel_shape[2];
        if (th.kernel_shape[1] == 1) {

            for (int i = 0; i < th.kernel_shape[0]; i++) {
                for (int j = 0; j < th.kernel_shape[3]; j++) {
                    for (int k = 0; k < th.kernel_shape[2]; k++) {

                        if (abs(th.convolution_kernels[i][0][k][j]) < eps) continue;
                        int cur_color = channel_colors[min(channel_colors.length - 1, k)];
                        if (th.kernel_shape[2] != channel_colors.length)
                            cur_color = channel_colors[min(channel_colors.length - 1, i)];

                        float new_red = constrain(weight_scalar * red(cur_color) * th.convolution_kernels[i][0][k][j], 0, 255);
                        float new_blue = constrain(weight_scalar * blue(cur_color) * th.convolution_kernels[i][0][k][j], 0, 255);
                        float new_green = constrain(weight_scalar * green(cur_color) * th.convolution_kernels[i][0][k][j], 0, 255);
                        float alpha = constrain(255.0f * weight_scalar * th.convolution_kernels[i][0][k][j], 0, 255);
                        int my_color = color(new_red, new_green, new_blue, alpha);

                        fill(my_color);
                        pushMatrix();
                        translate(-sizer / 2 + (float) i * x_inc, -sizer / 2 + (float) j * y_inc, -sizer / 2 + (float) k * z_inc);
                        box(x_inc, y_inc, z_inc);
                        popMatrix();

                    }
                }
            }
        } else if (th.kernel_shape[0] == 1) {
            x_inc = sizer / (float) th.kernel_shape[1];
            for (int i = 0; i < th.kernel_shape[1]; i++) {
                for (int j = 0; j < th.kernel_shape[3]; j++) {
                    for (int k = 0; k < th.kernel_shape[2]; k++) {

                        if (abs(th.convolution_kernels[0][i][k][j]) < eps) continue;
                        int cur_color = channel_colors[min(channel_colors.length - 1, i)];

                        float new_red = constrain(weight_scalar * red(cur_color) * th.convolution_kernels[0][i][k][j], 0, 255);
                        float new_blue = constrain(weight_scalar * blue(cur_color) * th.convolution_kernels[0][i][k][j], 0, 255);
                        float new_green = constrain(weight_scalar * green(cur_color) * th.convolution_kernels[0][i][k][j], 0, 255);
                        float alpha = constrain(255.0f * weight_scalar * th.convolution_kernels[0][i][k][j], 0, 255);
                        int my_color = color(new_red, new_green, new_blue, alpha);

                        fill(my_color);
                        pushMatrix();
                        translate(-sizer / 2 + (float) i * x_inc, -sizer / 2 + (float) j * y_inc, -sizer / 2 + (float) k * z_inc);
                        box(x_inc, y_inc, z_inc);
                        popMatrix();


                    }
                }
            }
        }
    }
    private boolean hasReads(int depth){
        float nonRefSum = 0.0f;
        for (int j = 0; j < tensor_shape[channel_idx]; j++) {
            for (int k = 0; k < tensor_shape[seq_idx]; k++) {

                if (j > 4) continue;

                if (channel_idx == 0) {
                    nonRefSum += abs(th.read_tensor[j][depth][k]);
                } else if(channel_idx ==2){
                    nonRefSum += abs(th.read_tensor[depth][k][j]);

                }
            }
        }
        return nonRefSum > 0.0;

    }

    public void showTensor(){
        float eps = 0.01f;
        float sizer = 2.0f;
        float smaller = 0.1f;
        float x_inc = sizer / (float)tensor_shape[0];
        float y_inc = sizer / (float)tensor_shape[1];
        float z_inc = sizer / (float)tensor_shape[2];

        if (channel_idx == 0){
            x_inc *= smaller;
        } else if(channel_idx == 2){
            z_inc *= smaller;
            rotateX(PI/2.0f);
        }
        for (float i = 0; i < tensor_shape[0]; i++) {
            for (float j = 0; j < tensor_shape[1]; j++) {


                for (float k = 0; k < tensor_shape[2]; k++) {

                    if (abs(th.read_tensor[(int)i][(int)j][(int)k] ) < eps) continue;


                    int color_idx = 0;
                    if (channel_idx == 0){
                        if (!hasReads((int)j)) continue;
                        color_idx =  (int)i;
                    }else if (channel_idx == 2){
                        if (!hasReads((int)i)) continue;
                        color_idx =  (int)k;
                    }


                    float new_red = constrain(red(channel_colors[color_idx]) * th.read_tensor[(int)i][(int)j][(int)k], 0, 255);
                    float new_blue = constrain(blue(channel_colors[color_idx]) * th.read_tensor[(int)i][(int)j][(int)k], 0, 255);
                    float new_green = constrain(green(channel_colors[color_idx]) * th.read_tensor[(int)i][(int)j][(int)k], 0, 255);
                    float alpha = constrain( 255.0f*th.read_tensor[(int)i][(int)j][(int)k], 0, 255);
                    int my_color = color(new_red, new_green, new_blue, alpha);

                    if(highlight_channel && i != cur_channel){
                        my_color = color(new_red, new_green, new_blue, 35);
                    }
                    fill(my_color);
                    pushMatrix();
                    translate(-sizer/2+i*x_inc, -sizer/2+j*y_inc, -sizer/2+k*z_inc);
                    box(x_inc, y_inc, z_inc);
                    popMatrix();


                }
            }
        }
        if (calling_tensors) {
            for (float k = 0; k < tensor_shape[2]; k++) {
                fill(channel_colors[(int) th.label_tensor[(int) k]]);
                pushMatrix();
                translate((-sizer/2) -(x_inc*3), -sizer / 2, -sizer / 2 + k * z_inc);
                box(x_inc*20, y_inc, z_inc);
                popMatrix();
            }

        }

    }


    public void showDNALogos(float letter_scale){
        int logo_x = 30;
        int logo_y  = 100;
        int logo_width = 500;
        int logo_height = 100;
        int max_logos = 21;
        for(int i = 0; i < max_logos; i++){
            int kernel_idx = (cur_kernel + i) % th.dna_weights[0][0].length;
            showDNALogo(logo_x, logo_y, kernel_idx, letter_scale);
            logo_y += logo_height;
            if (logo_y >= height - 40) {
                logo_y = logo_height;
                logo_x += logo_width;
            }
        }
    }

    public void showDNALogo(int logo_x, int logo_y, int tensor_idx, float letter_scale){
        float weight_scale = 100.0f;
        for (int i = 0; i < th.dna_weights.length; i++) {
            float max_weight = abs(th.dna_weights[i][0][tensor_idx]);
            int max_j = 0;
            float softmax_denom = exp(weight_scale*abs(th.dna_weights[i][0][tensor_idx]));
            for (int j = 1; j < dna.length; j++) {
                float cur_abs_weight = abs(th.dna_weights[i][j][tensor_idx]);
                softmax_denom += exp(weight_scale*cur_abs_weight);
                if (max_weight < cur_abs_weight){
                    max_weight = cur_abs_weight;
                    max_j = j;
                }
            }

            float y_pos = logo_y;
            float y_neg = logo_y;
            double p_max = exp(weight_scale*abs(th.dna_weights[i][max_j][tensor_idx])) / softmax_denom;
            double scalar = letter_scale * -p_max * Math.log(p_max); //exp(max_weight) / softmax_denom;

            for (int j = 0; j < dna.length; j++) {
                if(max_logo_only && j != max_j) continue;

                double p_j = exp(weight_scale*abs(th.dna_weights[i][j][tensor_idx])) / softmax_denom;
                double my_scalar = 0.001 + letter_scale * -p_j * Math.log(p_j);

                textSize((float)my_scalar);
                fill(channel_colors[j]);
                if (th.dna_weights[i][j][tensor_idx] < 0) {
                    pushMatrix();

                    translate(logo_x, y_neg, 0);
                    rotateZ(-PI);
                    translate(-(float)my_scalar/1.5f,0,0);
                    text(dna[j], 0, 0);
                    popMatrix();
                    y_neg += my_scalar;
                } else {
                    text(dna[j], logo_x, y_pos);
                    y_pos -= my_scalar;
                }
            }
            logo_x += scalar;

        }


    }

    public void showDNAReference(float[][] seq, int logo_x, int logo_y, float letter_scale){
        for (int i = 0; i < seq.length; i++) {
            float max_weight = seq[i][0];
            int max_j = 0;
            for (int j = 1; j < dna.length; j++) {
                if (abs(max_weight) < abs(seq[i][j])){
                    max_weight = seq[i][j];
                    max_j = j;
                }
            }
            float scalar = max(1.0f, abs(max_weight*letter_scale));
            textSize(scalar);
            fill(channel_colors[max_j]);
            if (max_weight < 0){
                pushMatrix();
                translate(logo_x, logo_y, 0);
                rotateZ(PI);
                rotateY(-PI);
                text(dna[max_j], 0, 0);
                popMatrix();
            } else {
                text(dna[max_j], logo_x, logo_y);
            }
            logo_x += scalar;
            // println("Show dna:", logo_x, max_weight);
        }


    }


}
