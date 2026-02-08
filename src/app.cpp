#include "app.hpp"

int run(int argc, char** argv)
{
    if (argc < 3)
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string op = argv[1];

    if (op == "split")
    {
        return cmd_split(argv[2], argv[3]);
    }
    if (op == "train" || op == "predict")
    {
        if (op == "train")
            return cmd_train(argv[2]);
        else if (argc == 4)
            return cmd_predict(argv[2], argv[3]);

        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::cerr << "Error: unknown operation '" << op << "'\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
}
