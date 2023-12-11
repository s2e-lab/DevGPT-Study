import ogr
import threading
import proj

def worker_function():
    # Create a new Proj context for each thread
    with proj.ProjContext() as ctx:
        # Set the Proj context on the GDAL/OGR library
        ogr.UseExceptions()
        ogr.SetDefaultContext(ctx)

        # Create separate source and target spatial reference systems
        source_srs = ogr.osr.SpatialReference()
        target_srs = ogr.osr.SpatialReference()
        source_srs.ImportFromEPSG(4326)
        target_srs.ImportFromEPSG(3857)

        # Create a new OGRCoordinateTransformation object
        transformation = ogr.osr.CoordinateTransformation(source_srs, target_srs)

        # Use the transformation object safely

# Create multiple threads
num_threads = 4
threads = []
for _ in range(num_threads):
    t = threading.Thread(target=worker_function)
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()
