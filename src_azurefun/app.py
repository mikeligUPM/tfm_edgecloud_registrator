from flask import Flask, request, render_template_string
import open3d as o3d
from visualizer_blob_downloader import *

app = Flask(__name__)

# Enable WebRTC for Open3D
o3d.visualization.webrtc_server.enable_webrtc()

html_template = '''
<!doctype html>
<html>
<head>
    <title>PLY File Viewer</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script type="text/javascript">
        $(document).ready(function() {
            $('form').on('submit', function(event) {
                event.preventDefault();
                $.ajax({
                    type: 'POST',
                    url: '/',
                    data: $(this).serialize(),
                    success: function(response) {
                        $('#result').html(response);
                    },
                    error: function() {
                        $('#result').html('An error occurred.');
                    }
                });
            });

            $('input[name="seq"]').on('change', function() {
                updateFrameOptions();
            });

            $('#blob-select').on('change', function() {
                var selectedBlob = $(this).val();
                if (selectedBlob && selectedBlob !== '----') {
                    $('input[name="reg_type"]').prop('disabled', true);
                    $('input[name="seq"]').prop('disabled', true);
                    $('#frame-options').prop('disabled', true);
                } else {
                    $('input[name="reg_type"]').prop('disabled', false);
                    $('input[name="seq"]').prop('disabled', false);
                    $('#frame-options').prop('disabled', false);
                }
            });

            function updateFrameOptions() {
                var dataset = $('input[name="seq"]:checked').val();
                var frameOptions = '<option value="" selected disabled>Select Frame</option>';
                
                if (dataset === '3DMatch') {
                    for (var i = 0; i <= 20; i++) {
                        var frame = 'frame-' + String(i).padStart(6, '0');
                        frameOptions += '<option value="' + frame + '">' + frame + '</option>';
                    }
                } else if (dataset === 'Own data') {
                    for (var i = 1; i <= 21; i++) {
                        var frame = 'f' + String(i).padStart(4, '0');
                        frameOptions += '<option value="' + frame + '">' + frame + '</option>';
                    }
                }
                $('#frame-options').html(frameOptions);
            }
        });
    </script>
</head>
<body>
    <h1>Enter Parameters to Load PLY File</h1>
    <div style="display: flex; justify-content: space-between;">
        <div>
            <form method="post">
                <h3>Registration Type:</h3>
                <label><input type="radio" name="reg_type" value="icp_p2p_ransac"> ICP P2P RANSAC</label><br>
                <label><input type="radio" name="reg_type" value="icp_p2l_ransac"> ICP P2L RANSAC</label><br>
                <label><input type="radio" name="reg_type" value="geotransformer"> GeoTransformer</label><br>
                
                <h3>Dataset:</h3>
                <label><input type="radio" name="seq" value="3DMatch"> 3DMatch</label><br>
                <label><input type="radio" name="seq" value="Own data"> Own Data</label><br>
                
                <h3>Frame:</h3>
                <select name="frame" id="frame-options">
                    <option value="" selected disabled>Select Frame</option>
                    <!-- Frame options will be populated based on dataset selection -->
                </select>
                
                <h3>Select Blob:</h3>
                <select id="blob-select" name="blob">
                    <option value="----" selected>----</option>
                    <!-- Blob options will be populated here -->
                    <!-- For example: -->
                    <option value="blob1.ply">blob1.ply</option>
                    <option value="blob2.ply">blob2.ply</option>
                    <!-- Add more options dynamically based on available blobs -->
                </select>
                
                <input type="submit" value="Load and Visualize">
            </form>
        </div>
    </div>
    <div id="result"></div>
</body>
</html>


'''  # Use the updated HTML template here

@app.route('/', methods=['GET', 'POST'])
def index():
    blob_service_client = get_blob_service_client_connection_string()
    blob_list = list_blobs_sorted(blob_service_client, 'cameraframes')
    blob_options = '<option value="----" selected>----</option>'
    blob_options += ''.join([f'<option value="{blob}">{blob}</option>' for blob in blob_list])
    
    if request.method == 'POST':
        reg_type = request.form.get('reg_type')
        frame = request.form.get('frame')
        blob = request.form.get('blob')
        
        filename = None
        if blob and blob != '----':
            filename = download_blob(blob_service_client, blob)
        elif reg_type and frame:
            filename = find_target_blob(frame, reg_type)
        
        if filename:
            try:
                pcd = o3d.io.read_point_cloud(filename)
                o3d.visualization.draw_geometries([pcd])
                return f"Loaded and visualized: {filename}"
            except Exception as e:
                return f"Error loading file {filename}: {e}"
        else:
            return "Please provide all parameters."
    
    # On GET request, render the form and populate blob options
    return render_template_string(html_template.replace('<!-- Blob options will be populated here -->', blob_options))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
