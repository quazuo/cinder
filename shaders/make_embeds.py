import sys
import os

if __name__ == "__main__":
    shader_filename = os.path.basename(sys.argv[1])

    with open("obj/" + shader_filename, "rb") as f:
        shader_bytes = f.read()

    with open("embeds/" + shader_filename, "w+") as f:
        embed = """
namespace zrx::spirv_embeds {{
    static constexpr std::array<uint32_t, {n_bytes}> {arr_name} {{ {bytes} }};
}}
        """.format(
            n_bytes = len(shader_bytes),
            arr_name=shader_filename.split(".")[0].replace("-", "_"),
            bytes = ", ".join(["0x" + shader_bytes[i : i + 4].hex().upper() for i in range(0, len(shader_bytes), 4)])
        )

        f.write(embed)