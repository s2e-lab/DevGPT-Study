import argparse

def main():
    parser = argparse.ArgumentParser(description="Select points 2D Image Selector")
    parser.add_argument("image", help="Image file path")
    parser.add_argument("--title", "-t", help="Window title", default="2D Image Selector")
    parser.add_argument("--points", "-p", type=int, default=4, help="Number of points to select")
    parser.add_argument("--show-outline", "-s", action="store_true", help="Show the outline of the quadrilateral")
    parser.add_argument("--closed", "-c", action="store_true", help="Indicate if the drawn path is closed")
    args = parser.parse_args()

    # Pass the title as an optional argument
    app = Select2DApp(title=args.title, num_points=args.points, show_outline=args.show_outline, closed=args.closed)
    app.load_image(args.image)
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()

if __name__ == '__main__':
    main()
