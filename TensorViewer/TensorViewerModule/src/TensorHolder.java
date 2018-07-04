import hdf.hdf5lib.H5;
import hdf.hdf5lib.HDF5Constants;
import hdf.hdf5lib.exceptions.HDF5Exception;

import java.nio.file.Path;
import java.util.Arrays;

/**
 * Created by sam on 8/19/17.
 */
public class TensorHolder {
    float[][] reference_tensor;
    float[][][] read_tensor;
    float[][][] dna_weights;
    float[] label_tensor;


    float[][][][] convolution_kernels;
    float[] convolution_bias;
    long[] kernel_shape;

    float[] bias;
    float[] annotations;
    String[] annotation_names;


    public TensorHolder(String[] annotation_names) {
        this.annotation_names = annotation_names;
    }


    private void scale_tensor(){
        float maxv = -9e9f;
        float minv = 9e9f;
        float eps = 1e-7f;
        for (int i = 0; i < read_tensor.length; i++) {
            for (int j = 0; j < read_tensor[0].length; j++) {
                for (int k = 0; k < read_tensor[0][0].length; k++) {
                    maxv = Math.max(read_tensor[i][j][k], maxv);
                    minv = Math.min(read_tensor[i][j][k], minv);
                }
            }
        }

        for (int i = 0; i < read_tensor.length; i++) {
            for (int j = 0; j < read_tensor[0].length; j++) {
                for (int k = 0; k < read_tensor[0][0].length; k++) {
                    read_tensor[i][j][k] -= minv;
                    read_tensor[i][j][k] /= (eps+maxv-minv);
                }
            }
        }
    }


    public void load_tensor_3d(Path tensor_file, String dataset_name, int[] tensor_shape){
        int file_id = -1;
        int dataset_id = -1;

        // Open file using the default properties.
        try {
            file_id = H5.H5Fopen(tensor_file.toString(), HDF5Constants.H5F_ACC_RDWR, HDF5Constants.H5P_DEFAULT);
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        // Open dataset using the default properties.
        try {
            if (file_id >= 0) {
                dataset_id = H5.H5Dopen(file_id, dataset_name, HDF5Constants.H5P_DEFAULT);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        // Allocate 3D array
        read_tensor = new float[tensor_shape[0]][tensor_shape[1]][tensor_shape[2]];

        try {
            if (dataset_id >= 0) {
                H5.H5Dread(dataset_id, HDF5Constants.H5T_NATIVE_FLOAT,
                        HDF5Constants.H5S_ALL, HDF5Constants.H5S_ALL,
                        HDF5Constants.H5P_DEFAULT, read_tensor);
                System.out.println("Tensor loaded. Shape:" + Arrays.toString(tensor_shape));
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void load_label_tensor(Path tensor_file, String tensor_name, int window_size){
        int file_id = -1;
        int data_id = -1;

        // Open file using the default properties.
        try {
            file_id = H5.H5Fopen(tensor_file.toString(), HDF5Constants.H5F_ACC_RDWR, HDF5Constants.H5P_DEFAULT);
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        // Open dataset using the default properties.
        try {
            if (file_id >= 0) {
                data_id = H5.H5Dopen(file_id, tensor_name, HDF5Constants.H5P_DEFAULT);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        label_tensor = new float[window_size];

        if(data_id >= 0){
            try {
                H5.H5Dread(data_id, HDF5Constants.H5T_NATIVE_FLOAT,
                        HDF5Constants.H5S_ALL, HDF5Constants.H5S_ALL,
                        HDF5Constants.H5P_DEFAULT, label_tensor);
                System.out.println(tensor_name+" labels loaded.");
            } catch (HDF5Exception e) {
                e.printStackTrace();
            }
        }
    }


    public void load_annotations(String annotation_set, int num_annotations, Path tensor_file){
        int file_id = -1;
        int annotation_data_id = -1;

        // Open file using the default properties.
        try {
            file_id = H5.H5Fopen(tensor_file.toString(), HDF5Constants.H5F_ACC_RDWR, HDF5Constants.H5P_DEFAULT);
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        // Open dataset using the default properties.
        try {
            if (file_id >= 0) {
                annotation_data_id = H5.H5Dopen(file_id, annotation_set, HDF5Constants.H5P_DEFAULT);

            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        annotations = new float[num_annotations];

        if(annotation_data_id >= 0){
            try {
                H5.H5Dread(annotation_data_id, HDF5Constants.H5T_NATIVE_FLOAT,
                        HDF5Constants.H5S_ALL, HDF5Constants.H5S_ALL,
                        HDF5Constants.H5P_DEFAULT, annotations);
                System.out.println(annotation_set + " set of annotations loaded.");
            } catch (HDF5Exception e) {
                e.printStackTrace();
            }
        }
    }

    public void load_reference_tensor(Path tensor_file, String dataset_name, int[] reference_shape){
        int file_id = -1;
        int dataset_id = -1;

        // Open file using the default properties.
        try {
            file_id = H5.H5Fopen(tensor_file.toString(), HDF5Constants.H5F_ACC_RDWR, HDF5Constants.H5P_DEFAULT);
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        // Open dataset using the default properties.
        try {
            if (file_id >= 0) {
                dataset_id = H5.H5Dopen(file_id, dataset_name, HDF5Constants.H5P_DEFAULT);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        // Allocate 2D array
        reference_tensor =  new float[reference_shape[0]][reference_shape[1]];
        try {
            if (dataset_id >= 0) {
                H5.H5Dread(dataset_id, HDF5Constants.H5T_NATIVE_FLOAT,
                        HDF5Constants.H5S_ALL, HDF5Constants.H5S_ALL,
                        HDF5Constants.H5P_DEFAULT, reference_tensor);
                System.out.println("Reference Tensor loaded." + Arrays.toString(reference_tensor[0]));

            }

        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }


    public void load_convolution_kernels(String weights_path, String layer_name) {
        int file_id = -1;
        int dataset_id = -1;

        // Open file using the default properties.
        try {
            file_id = H5.H5Fopen(weights_path, HDF5Constants.H5F_ACC_RDWR, HDF5Constants.H5P_DEFAULT);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Open data set using the default properties.
        try {
            if (file_id >= 0) {
                dataset_id = H5.H5Gopen(file_id, "model_weights", HDF5Constants.H5P_DEFAULT);
                int aid = H5.H5Gopen(dataset_id, layer_name, HDF5Constants.H5P_DEFAULT);
                int aid2 = H5.H5Gopen(aid, layer_name, HDF5Constants.H5P_DEFAULT);
                int aid3 = H5.H5Dopen(aid2, "kernel",HDF5Constants.H5P_DEFAULT);
                int space_id = H5.H5Dget_space(aid3);
                int aid4 = H5.H5Dopen(aid2, "bias", HDF5Constants.H5P_DEFAULT);
                long[] max_dims = new long[4];
                kernel_shape = new long[4];
                int num_d = H5.H5Sget_simple_extent_dims(space_id, kernel_shape, max_dims);
                int dataspace = H5.H5Dget_space(aid3);

                System.out.println("data set rank:"+ num_d + "  " + Arrays.toString(kernel_shape)+ "  " + Arrays.toString(max_dims));
                if (num_d == 4){
                    convolution_kernels = new float[(int)kernel_shape[0]][(int)kernel_shape[1]][(int)kernel_shape[2]][(int)kernel_shape[3]];
                    float[] kernels2 = new float[(int)(kernel_shape[0]*kernel_shape[1]*kernel_shape[2]*kernel_shape[3])];
                    H5.H5Dread_float(aid3, HDF5Constants.H5T_IEEE_F32LE, HDF5Constants.H5S_ALL, dataspace, HDF5Constants.H5P_DEFAULT, kernels2);
                    for (int i = 0; i < kernel_shape[0]; i++) {
                        for (int j = 0; j < kernel_shape[1]; j++) {
                            for (int k = 0; k < kernel_shape[2]; k++) {
                                for (int l = 0; l < kernel_shape[3]; l++) {
                                    int index = l + (int)( k*kernel_shape[3] + (j*kernel_shape[2]*kernel_shape[3]) + (i*kernel_shape[1]*kernel_shape[2]*kernel_shape[3]) );
                                    convolution_kernels[i][j][k][l] = kernels2[index];
                                }
                            }
                        }
                    }

                    System.out.println("Loaded Kernel Shape:  " + Arrays.toString(kernel_shape));

                    convolution_bias = new float[(int)kernel_shape[3]];
                    H5.H5Dread(aid4, HDF5Constants.H5T_IEEE_F32LE,
                            HDF5Constants.H5S_ALL, HDF5Constants.H5S_ALL,
                            HDF5Constants.H5P_DEFAULT, convolution_bias);
                    System.out.println("Loaded Biases:  " + Arrays.toString(bias) + "\n selected "+kernels2[0] + "  last = " + kernels2[256*14]);
                }

            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }


    public void load_dna_weights(String weights_path, String layer_name) {
        int file_id = -1;
        int dataset_id = -1;

        // Open file using the default properties.
        try {
            file_id = H5.H5Fopen(weights_path, HDF5Constants.H5F_ACC_RDWR, HDF5Constants.H5P_DEFAULT);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Open data set using the default properties.
        try {
            if (file_id >= 0) {
                dataset_id = H5.H5Gopen(file_id, "model_weights", HDF5Constants.H5P_DEFAULT);
                int aid = H5.H5Gopen(dataset_id, layer_name, HDF5Constants.H5P_DEFAULT);
                int aid2 = H5.H5Gopen(aid, layer_name, HDF5Constants.H5P_DEFAULT);
                int aid3 = H5.H5Dopen(aid2, "kernel",HDF5Constants.H5P_DEFAULT);
                int space_id = H5.H5Dget_space(aid3);
                int aid4 = H5.H5Dopen(aid2, "bias", HDF5Constants.H5P_DEFAULT);
                long[] max_dims = new long[3];
                long[] kernel_shape = new long[3];
                int num_d = H5.H5Sget_simple_extent_dims(space_id, kernel_shape, max_dims);
                int dataspace = H5.H5Dget_space(aid3);

                dna_weights = new float[(int)kernel_shape[0]][(int)kernel_shape[1]][(int)kernel_shape[2]];
                System.out.println("data set rank:"+ num_d + "  " + Arrays.toString(kernel_shape)+ "  " + Arrays.toString(max_dims));
                assert(num_d == 3);

                float[] dna_tensors2 = new float[(int) (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])];
                H5.H5Dread_float(aid3, HDF5Constants.H5T_IEEE_F32LE, HDF5Constants.H5S_ALL, dataspace, HDF5Constants.H5P_DEFAULT, dna_tensors2);
                for (int i = 0; i < kernel_shape[0]; i++) {
                    for (int j = 0; j < kernel_shape[1]; j++) {
                        for (int k = 0; k < kernel_shape[2]; k++) {
                            int index = (int) (k + (j * kernel_shape[2]) + (i * kernel_shape[1] * kernel_shape[2]));
                            dna_weights[i][j][k] = dna_tensors2[index];
                        }
                    }
                }
                System.out.println("Loaded Kernel:  " + Arrays.toString(dna_weights[0][0]));

                float[] dna_biases = new float[(int) kernel_shape[2]];
                H5.H5Dread(aid4, HDF5Constants.H5T_IEEE_F32LE,
                        HDF5Constants.H5S_ALL, HDF5Constants.H5S_ALL,
                        HDF5Constants.H5P_DEFAULT, dna_biases);
                System.out.println("Loaded Biases:  " + Arrays.toString(dna_biases) + "\n selected " + dna_biases[0]);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
}
