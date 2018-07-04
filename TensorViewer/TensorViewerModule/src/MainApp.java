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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;


public class MainApp extends PApplet {
    private static String fname  = "/dsde/working/sam/palantir_cnn/Analysis/vqsr_cnn/generated_1d/excite_conv1d_2/filter_22.hd5";

    int cur_dataset = 0;
    String data_root = "/dsde/data/deep/vqsr/tensors/";
    private String[] datasets = {
            data_root+"g947j_paired_read2/train",data_root+"g947j_baseline2/train",
            "/dsde/working/sam/dsde-deep-learning/weights/g947_site_labelled_rrab3/",
            data_root+"g94982_calling_channels_last/", data_root+"g94982_na12878_ref_read_anno_channels_first/test",
            data_root+"g947_paired_read2/train", data_root+"g947u_paired_read/train",
            data_root+"g94982_channels_first_calling_tensors_het_enrich/",
            data_root+"g94982_calling_tensors_indel_only/", data_root+"g94982_calling_tensors_variant_sort/"
    };
    boolean calling_tensors = false;

    private static String weights_path = "/dsde/working/sam/palantir_cnn/Analysis/vqsr_cnn/weights/m__base_quality_mode_phot__channels_last_False__id_g94982_mq_train__window_size_128__read_limit_128__random_seed_12878__tensor_map_2d_mapping_quality__mode_ref_read_anno.hd5";

    private static String dna_weights_path = "/dsde/working/sam/dsde-deep-learning/weights/dna_weights/new_1layer_spatial_drop_epoch_361.hd5";
    //private static String dna_weights_path = "/dsde/working/sam/palantir_cnn/Analysis/vqsr_cnn/weights/1d_ref_anno_skip2.hd5";
    int movie_frames = 362;
    int cur_label = 0;
    int cur_tensor = 0;
    float cur_scale = 100;
    float max_scale = 1080;
    float min_scale = 10;

    List<List<Path>> tensor_files;


    int num_channels = 15;
    int window_size = 128;
    int read_limit = 128;

    //int[] tensor_shape = {read_limit, window_size, num_channels};
    int[] tensor_shape = {num_channels, read_limit, window_size};
    int channel_idx = 0;

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
    int seq_idx = 1;
    int cur_kernel = 92;
    int cur_layer_idx =  0;
    String[] layer_names = {"conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4"};
    int samples = 10;
    String annotation_set = "best_practices";
    String[] annotation_names = {"MQ", "DP", "SOR", "FS", "QD", "MQRankSum", "ReadPosRankSum"};
    boolean load_annotations = true;
    int cur_mode = 0;
    boolean highlight_channel = false;
    boolean max_logo_only = false;
    String[] modes = {"tensor", "weights", "logos", "reference"};
    TensorHolder th;

    float xmag, ymag = 0;
    float newXmag, newYmag = 0;

    public static void main(String[] args){
        PApplet.main("MainApp", args);
    }

    public void settings(){
        size(1600, 1200,  "processing.opengl.PGraphics3D");
        tensor_files = getTensorFiles(datasets[cur_dataset]);
        th = new TensorHolder(annotation_names);
        th.load_tensor_3d(tensor_files.get(cur_label).get(cur_tensor), "read_tensor", tensor_shape);
        th.load_dna_weights(dna_weights_path, "conv1d_1");
        th.load_reference_tensor(Paths.get(fname), "reference", reference_shape);
        th.load_convolution_kernels(weights_path, layer_names[cur_layer_idx]);
        if (calling_tensors){
            th.load_label_tensor(tensor_files.get(cur_label).get(cur_tensor), "site_labels", tensor_shape[1]);
        }
        if (load_annotations){
            th.load_annotations(annotation_set, annotation_names.length, tensor_files.get(cur_label).get(cur_tensor));
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
            textSize(20);
            text(datasets[cur_dataset], 220, 24);
            text(tensor_files.get(cur_label).get(cur_tensor).getFileName().toString(), 220, 48);
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
                text_y = height - 20;
                textSize(20);
                fill(20);
                for (int i = 0; i < annotation_names.length; i++) {
                    text_y -= text_line_height*0.7f;
                    text(annotation_names[i] + " : " + Float.toString(th.annotations[i]), width-270, text_y);
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
            //showDNALogos(cur_scale*0.95f, "medium");
            showDNALogos(cur_scale*2.8f, "single");

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
            cur_kernel = (1+cur_kernel) % th.dna_weights[0][0].length;
            println(" cur_kernel #:", cur_kernel);
        } else if (key =='D'){
            load_dataset();
        } else if (key =='c'){
            cur_channel = ++cur_channel % channel_names.length;
        } else if (key =='h'){
            highlight_channel = !highlight_channel;
            max_logo_only = !max_logo_only;
        } else if (key =='m'){
            cur_mode = ++cur_mode% modes.length;
        } else if (key =='M'){
            makeMovie();
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
            if (load_annotations){
                th.load_annotations(annotation_set, annotation_names.length, tensor_files.get(cur_label).get(cur_tensor));
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
                } else if(channel_idx == 2){
                    nonRefSum += abs(th.read_tensor[depth][k][j]);

                }
            }
        }
        return nonRefSum > 0.0;

    }

    public void showTensor(){
        float eps = 0.4f;
        float sizer = 2.0f;
        float smaller = 0.5f;
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
                    //float alpha = constrain( 255.0f*th.read_tensor[(int)i][(int)j][(int)k], 0, 255);
                    int my_color = color(new_red, new_green, new_blue);

                    if(highlight_channel){
                        if ((channel_idx == 0 && i != cur_channel) || (channel_idx == 2 && k != cur_channel)) {
                            //my_color = color(new_red, new_green, new_blue, 0);
                            continue;
                        }
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
            for (float k = 0; k < tensor_shape[2-channel_idx]; k++) {
                fill(channel_colors[(int) th.label_tensor[(int) k]]);
                pushMatrix();
                if (channel_idx == 0) {
                    translate((-sizer / 2) - (x_inc * 3), -sizer / 2, -sizer / 2 + k * z_inc);
                    box(x_inc * 30, y_inc, z_inc);
                } else if (channel_idx == 2){
                    translate((-sizer / 2)- (x_inc * 3), -sizer/2+k * y_inc, -(sizer / 2) );
                    box(x_inc, y_inc, z_inc*30);
                }

                if((int) th.label_tensor[(int)k] > 0 && (k == 0 || (int)th.label_tensor[(int)(k-1)] == 0)) {
                    textSize(0.07f);
                    rotateX(PI);
                    if (channel_idx == 0) {
                        rotateY(-PI/2);
                        text(calling_labels[(int) th.label_tensor[(int) k]], x_inc, -2*y_inc, k*z_inc/2);
                    } else if (channel_idx == 2){
                        rotateZ(PI/2);
                        text(calling_labels[(int) th.label_tensor[(int) k]], x_inc*3, -y_inc*3, k*x_inc/2);
                    }

                }
                popMatrix();
            }

        }

    }


    public void showDNALogos(float letter_scale, String mode){
        int logo_x = 20;
        int logo_y_start = 90;
        int logo_y = logo_y_start;
        int logo_width = 115;
        int logo_height = 55;
        int max_logos = 512;

        if (mode.equalsIgnoreCase("medium")) {
            logo_x = 30;
            logo_y_start = 150;
            logo_y = logo_y_start;
            logo_width = 550;
            logo_height = 170;
            max_logos = 21;
        } else if (mode.equalsIgnoreCase("single")) {
            logo_x = 50;
            logo_y_start = 450;
            logo_y = logo_y_start;
            logo_width = 1450;
            logo_height = 360;
            max_logos = 1;
        } 
        
        for(int i = 0; i < max_logos; i++){
            int kernel_idx = (cur_kernel + i) % th.dna_weights[0][0].length;
            showDNALogo(logo_x, logo_y, kernel_idx, letter_scale);
            logo_y += logo_height;
            if (logo_y >= height - logo_height) {
                logo_y = logo_y_start;
                logo_x += logo_width;
            }
        }
    }

    public void showDNALogo(int logo_x, int logo_y, int tensor_idx, float letter_scale){
        float weight_scale = 18.0f;
        float min_height = 0.002f;
        int logo_x_start = logo_x;
        for (int i = 0; i < th.dna_weights.length; i++) {
            float max_weight = abs(th.dna_weights[i][0][tensor_idx]);
            int max_j = 0;
            float softmax_denom = exp(weight_scale*max_weight);
            for (int j = 1; j < dna.length; j++) {
                float cur_abs_weight = abs(th.dna_weights[i][j][tensor_idx]);
                softmax_denom += exp(weight_scale*cur_abs_weight);
                if (max_weight < cur_abs_weight){
                    max_weight = cur_abs_weight;
                    max_j = j;
                }
            }

            float y_pos = logo_y-3;
            float y_neg = logo_y+3;

            Pair[] sortedDNAChars = new Pair[4];
            for (int j = 0; j < dna.length; j++) {
                sortedDNAChars[j] = new Pair(j, th.dna_weights[i][j][tensor_idx]);
            }
            Arrays.sort(sortedDNAChars);

            for (int j = 0; j < sortedDNAChars.length; j++) {
                if (max_logo_only && sortedDNAChars[j].index != max_j) continue;
                if (sortedDNAChars[j].value < 0) continue;
                double p_j = exp(weight_scale*sortedDNAChars[j].value) / softmax_denom;
                double my_scalar = min_height + (letter_scale * p_j);

                textSize((float) my_scalar);
                fill(channel_colors[sortedDNAChars[j].index]);
                text(dna[sortedDNAChars[j].index], logo_x, y_pos);
                y_pos -= my_scalar;
            }

            for (int j = sortedDNAChars.length-1; j >= 0; j--) {
                if (max_logo_only && sortedDNAChars[j].index != max_j) continue;
                if (sortedDNAChars[j].value >= 0) continue;
                double p_j = exp(weight_scale*abs(sortedDNAChars[j].value)) / softmax_denom;
                double my_scalar = min_height + (letter_scale * p_j );

                textSize((float) my_scalar);
                fill(channel_colors[sortedDNAChars[j].index]);
                pushMatrix();
                translate(logo_x, y_neg, 0);
                rotateZ(-PI);
                translate(-(float) my_scalar / 1.5f, 0, 0);
                text(dna[sortedDNAChars[j].index], 0, 0);
                popMatrix();
                y_neg += my_scalar;
            }

            double p_max = exp(weight_scale*abs(th.dna_weights[i][max_j][tensor_idx])) / softmax_denom;
            double scalar = (letter_scale * p_max );
            logo_x += scalar;

        }
        stroke(50);
        line(logo_x_start, logo_y, logo_x, logo_y);
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

    public void makeMovie(){
        for (int i = 0; i < movie_frames; i++) {
            String weights_path = "/dsde/working/sam/dsde-deep-learning/weights/dna_weights/1layer_spatial_drop_epoch_" + String.valueOf(i) + ".hd5";
            th.load_dna_weights(weights_path, "conv1d_1");
            showDNALogos(cur_scale*2.8f, "single");
            //showDNALogos(cur_scale*0.95f, "medium");
            //showDNALogos(cur_scale*0.26f, "grid");

            save("/Users/sam/Desktop/dna_kernel_"+String.valueOf(cur_kernel)+"_global_"+String.valueOf(i)+".png");
            background(255);

        }
    }

    private void load_dataset(){
        cur_dataset = ++cur_dataset % datasets.length;
        tensor_files = getTensorFiles(datasets[cur_dataset]);
        cur_tensor = 0;
        cur_label = 0;
        calling_tensors = datasets[cur_dataset].toLowerCase().contains("calling");
        if (datasets[cur_dataset].toLowerCase().contains("channels_last")){
            tensor_shape[0] = read_limit;
            tensor_shape[1] = window_size;
            tensor_shape[2] = num_channels;
            channel_idx = 2;
        } else{
            tensor_shape[0] = num_channels;
            tensor_shape[1] = read_limit;
            tensor_shape[2] = window_size;
            channel_idx = 0;
        }
        th.load_tensor_3d(tensor_files.get(cur_label).get(cur_tensor), "read_tensor", tensor_shape);
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
                Collections.sort(label_tensors);
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

    public class Pair implements Comparable<Pair> {
        public final int index;
        public final float value;

        public Pair(int index, float value) {
            this.index = index;
            this.value = value;
        }

        @Override
        public int compareTo(Pair other) {
            //multiplied to -1 as the author need descending sort order
            return -1*Float.valueOf(this.value).compareTo(other.value);
        }
    }

}


