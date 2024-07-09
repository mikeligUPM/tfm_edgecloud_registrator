


from flask import Flask, request, render_template_string
import open3d as o3d

from visualizer_blob_downloader import find_target_blob

app = Flask(__name__)

# Enable WebRTC for Open3D
o3d.visualization.webrtc_server.enable_webrtc()

# HTML template for the form
# html_template = '''
# <!doctype html>
# <html>
# <head>
#     <title>PLY File Viewer</title>
# </head>
# <body>
#     <h1>Enter Parameters to Load PLY File</h1>
#     <form method="post">
#         registration type: <input type="text" name="reg_type"><br>
#         dataset: <input type="text" name="seq"><br>
#         frame: <input type="text" name="frame"><br>
#         <input type="submit" value="Load and Visualize">
#     </form>
# </body>
# </html>
# '''

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
                        $('#result').html(response.message);
                    },
                    error: function() {
                        $('#result').html('An error occurred.');
                    }
                });
            });

            $('input[name="seq"]').on('change', function() {
                updateFrameOptions();
            });

            function updateFrameOptions() {
                var dataset = $('input[name="seq"]:checked').val();
                var frameOptions = '';
                if (dataset === '3DMatch') {
                    for (var i = 0; i <= 20; i++) {
                        var frame = 'frame-' + String(i).padStart(6, '0');
                        frameOptions += '<label><input type="radio" name="frame" value="' + frame + '"> ' + frame + '</label><br>';
                    }
                } else if (dataset === 'Own data') {
                    for (var i = 1; i <= 21; i++) {
                        var frame = 'f' + String(i).padStart(4, '0');
                        frameOptions += '<label><input type="radio" name="frame" value="' + frame + '"> ' + frame + '</label><br>';
                    }
                }
                $('#frame-options').html(frameOptions);
            }
        });
    </script>
</head>
<body>
    <h1>Enter Parameters to Load PLY File</h1>
    <form>
        <h3>Registration Type:</h3>
        <label><input type="radio" name="reg_type" value="icp_p2p_ransac"> ICP P2P RANSAC</label><br>
        <label><input type="radio" name="reg_type" value="icp_p2l_ransac"> ICP P2L RANSAC</label><br>
        <label><input type="radio" name="reg_type" value="geotransformer"> GeoTransformer</label><br>
        
        <h3>Dataset:</h3>
        <label><input type="radio" name="seq" value="3DMatch"> 3DMatch</label><br>
        <label><input type="radio" name="seq" value="Own data"> Own Data</label><br>
        
        <h3>Frame:</h3>
        <div id="frame-options">
            <!-- Frame options will be populated based on dataset selection -->
        </div>
        
        <input type="submit" value="Load and Visualize">
    </form>
    <div id="result"></div>
</body>
</html>
'''


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        reg_type = request.form.get('reg_type')
        # seq = request.form.get('seq')
        frame = request.form.get('frame')
        
        if reg_type and frame:
            # o3d.visualization.webrtc_server.enable_webrtc()
            # filename = f"{reg_type}__{seq}__{frame}.ply"
            try:
                filename = find_target_blob(frame, reg_type)
                pcd = o3d.io.read_point_cloud(filename)
                o3d.visualization.draw_geometries([pcd])
                return f"Loaded and visualized: {filename}"
            except Exception as e:
                return f"Error loading file {filename}: {e}"
        else:
            return "Please provide all parameters."
    
    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
