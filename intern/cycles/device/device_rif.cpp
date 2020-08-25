/*
 * Copyright 2019, NVIDIA Corporation.
 * Copyright 2019, Blender Foundation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef WITH_RIF

#  include <boost/filesystem.hpp>

#  include "device/device.h"
#  include "device/device_intern.h"
#  include "device/opencl/device_opencl.h"

#  include "render/buffers.h"
#  include "render/hair.h"
#  include "render/mesh.h"
#  include "render/object.h"
#  include "render/scene.h"

#  include "util/util_debug.h"
#  include "util/util_logging.h"
#  include "util/util_md5.h"
#  include "util/util_path.h"
#  include "util/util_time.h"

#  include <RadeonImageFilters.h>
#  include <RadeonImageFilters_cl.h>

#  define RIF_DENOISER_NO_PIXEL_STRIDE 1

CCL_NAMESPACE_BEGIN

#  define check_result_rif(stmt) \
    { \
      rif_int res = stmt; \
      if (res != RIF_SUCCESS) { \
        const char *code = rifGetErrorCodeString(res); \
        const char *msg = rifGetLastErrorMessage(); \
        set_error( \
            string_printf("%s (%s) in %s (device_rif.cpp:%d)", msg, code, #stmt, __LINE__)); \
        return; \
      } \
    } \
    (void)0
#  define check_result_rif_ret(stmt) \
    { \
      rif_int res = stmt; \
      if (res != RIF_SUCCESS) { \
        const char *code = rifGetErrorCodeString(res); \
        const char *msg = rifGetLastErrorMessage(); \
        set_error( \
            string_printf("%s (%s) in %s (device_rif.cpp:%d)", msg, code, #stmt, __LINE__)); \
        return false; \
      } \
    } \
    (void)0

class RIFDevice : public OpenCLDevice {

  // Use a pool with multiple threads to support launches with multiple OpenCL queues
  TaskPool task_pool;

  template<typename T> class rif_object {
   private:
    T object = nullptr;
    rif_object(const rif_object &) = delete;             // non construction-copyable
    rif_object &operator=(const rif_object &) = delete;  // non copyable
   public:
    rif_object() = default;

    ~rif_object()
    {
      if (object)
        rifObjectDelete(object);
    }

    operator T()
    {
      return object;
    }

    T *operator&()
    {
      return &object;
    }
  };

  rif_object<rif_context> context;
  rif_object<rif_command_queue> queue;
  rif_object<rif_image_filter> denoise_filter;
  rif_object<rif_image_filter> remap_normal_filter;
  rif_object<rif_image_filter> remap_depth_filter;

  OpenCLProgram denoising_program;

 public:
  RIFDevice(DeviceInfo &info, Stats &stats, Profiler &profiler, bool background)
      : OpenCLDevice(info, stats, profiler, background)
  {
    // info.= DebugFlags().rif.;

    if (!cxContext) {
      return;  // Do not initialize if OpenCL context creation failed already
    }

    // Create RIF context
    auto &cache_path = boost::filesystem::temp_directory_path().string();
    check_result_rif(rifCreateContextFromOpenClContext(
        RIF_API_VERSION, cxContext, cdDevice, cqCommandQueue, cache_path.c_str(), &context));

    // Create RIF command queue
    check_result_rif(rifContextCreateCommandQueue(context, &queue));

    // Create RIF AI denoise filter
    check_result_rif(
        rifContextCreateImageFilter(context, RIF_IMAGE_FILTER_AI_DENOISE, &denoise_filter));
    check_result_rif(rifImageFilterSetParameter1u(denoise_filter, "useHDR", RIF_TRUE));

    // Create RIF remap filter for normals
    check_result_rif(
        rifContextCreateImageFilter(context, RIF_IMAGE_FILTER_REMAP_RANGE, &remap_normal_filter));
    check_result_rif(rifImageFilterSetParameter1f(remap_normal_filter, "dstLo", 0.0f));
    check_result_rif(rifImageFilterSetParameter1f(remap_normal_filter, "dstHi", 1.0f));

    // Create RIF remap depth filter
    check_result_rif(
        rifContextCreateImageFilter(context, RIF_IMAGE_FILTER_REMAP_RANGE, &remap_depth_filter));
    check_result_rif(rifImageFilterSetParameter1f(remap_depth_filter, "dstLo", 0.0f));
    check_result_rif(rifImageFilterSetParameter1f(remap_depth_filter, "dstHi", 1.0f));

    denoising_program = OpenCLDevice::OpenCLProgram(
        this,
        "denoising_program",
        "filter.cl",
        get_build_options(DeviceRequestedFeatures(), "denoising_program", false));

    denoising_program.add_kernel(ustring("filter_write_color"));
    denoising_program.add_kernel(ustring("filter_split_aov"));
    if (!denoising_program.load()) {
      denoising_program.compile();
    }
  }

  void thread_run(DeviceTask &task) override
  {
    flush_texture_buffers();

    if (task.type == DeviceTask::RENDER) {
      RenderTile tile;
      DenoisingTask denoising(this, task);

      /* Allocate buffer for kernel globals */
      //device_only_memory<KernelGlobalsDummy> kgbuffer(this, "kernel_globals");
      //kgbuffer.alloc_to_device(1);

      /* Keep rendering tiles until done. */
      while (task.acquire_tile(this, tile, task.tile_types)) {
        if (tile.task == RenderTile::PATH_TRACE) {
          assert(tile.task == RenderTile::PATH_TRACE);
          scoped_timer timer(&tile.buffers->render_time);

          //split_kernel->path_trace(task, tile, kgbuffer, *const_mem_map["__data"]);

          /* Complete kernel execution before release tile. */
          /* This helps in multi-device render;
           * The device that reaches the critical-section function
           * release_tile waits (stalling other devices from entering
           * release_tile) for all kernels to complete. If device1 (a
           * slow-render device) reaches release_tile first then it would
           * stall device2 (a fast-render device) from proceeding to render
           * next tile.
           */
          clFinish(cqCommandQueue);
        }
        else if (tile.task == RenderTile::BAKE) {
          bake(task, tile);
        }
        else if (tile.task == RenderTile::DENOISE) {
          tile.sample = tile.start_sample + tile.num_samples;
          launch_denoise(task, tile, denoising);
          task.update_progress(&tile, tile.w * tile.h);
        }

        task.release_tile(tile);
      }

      //kgbuffer.free();
    }
    else if (task.type == DeviceTask::SHADER) {
      shader(task);
    }
    else if (task.type == DeviceTask::FILM_CONVERT) {
      film_convert(task, task.buffer, task.rgba_byte, task.rgba_half);
    }
    else if (task.type == DeviceTask::DENOISE_BUFFER) {
      RenderTile tile;
      tile.x = task.x;
      tile.y = task.y;
      tile.w = task.w;
      tile.h = task.h;
      tile.buffer = task.buffer;
      tile.sample = task.sample + task.num_samples;
      tile.num_samples = task.num_samples;
      tile.start_sample = task.sample;
      tile.offset = task.offset;
      tile.stride = task.stride;
      tile.buffers = task.buffers;

      DenoisingTask denoising(this, task);
      launch_denoise(task, tile, denoising);
      task.update_progress(&tile, tile.w * tile.h);
    }
  }

 private:

  void merge_tiles(DeviceTask &task,
                   const float scale,
                   RenderTileNeighbors &neighbors,
                   array<float> &merged,
                   const int4 &rect,
                   const int2 &rect_size,
                   const size_t pass_stride)
  {
    /* Adjacent tiles are in separate memory regions, copy into single buffer. */
    merged.resize(size_t(rect_size.x * rect_size.y * task.pass_stride));

    //for (int i = 0; i < RenderTileNeighbors::SIZE; i++)
    {
      RenderTile &ntile = neighbors.tiles[RenderTileNeighbors::CENTER];
      //if (!ntile.buffer) {
      //  continue;
      //}

      ntile.buffers->copy_from_device();

      const int xmin = max(ntile.x, rect.x);
      const int ymin = max(ntile.y, rect.y);
      const int xmax = min(ntile.x + ntile.w, rect.z);
      const int ymax = min(ntile.y + ntile.h, rect.w);

      const size_t tile_offset = size_t(ntile.offset + xmin + ymin * ntile.stride);
      const float *tile_buffer = (float *)ntile.buffers->buffer.host_pointer + tile_offset * pass_stride;

      const size_t merged_stride = rect_size.x;
      const size_t merged_offset = (xmin - rect.x) + (ymin - rect.y) * merged_stride;
      float *merged_buffer = merged.data() + merged_offset * pass_stride;

      for (int y = ymin; y < ymax; y++) {
        for (int x = 0; x < pass_stride * (xmax - xmin); x++) {
          merged_buffer[x] = tile_buffer[x] * scale;
        }
        tile_buffer += ntile.stride * pass_stride;
        merged_buffer += merged_stride * pass_stride;
      }
    }
  }

  void split_aov(DeviceTask &task,
                 array<float> &buffer,
                 const int offset,
                 const int stride,
                 const int x,
                 const int y,
                 const int w,
                 const int h,
                 const float scale,
                 const int color_only,
                 device_vector<float> &color,
                 device_vector<float> &albedo,
                 device_vector<float> &normals,
                 device_vector<float> &depth)
  {
    /* Set images with appropriate stride for our interleaved pass storage. */
    const int pixel_offset = offset + x + y * stride;
    const int pixel_stride = task.pass_stride;
    const int row_stride = stride * pixel_stride;

    struct {
      device_vector<float> &vector;
      const int num_elements;
      const int offset;
      const bool scale;
      const bool use;
      array<float> scaled_buffer;
    } passes[] = {
        {color,
         3,
         pixel_offset * task.pass_stride + task.pass_denoising_data + DENOISING_PASS_COLOR,
         false,
         true},
        {albedo,
         3,
         pixel_offset * task.pass_stride + task.pass_denoising_data + DENOISING_PASS_ALBEDO,
         true,
         !color_only},
        {normals,
         3,
         pixel_offset * task.pass_stride + task.pass_denoising_data + DENOISING_PASS_NORMAL,
         true,
         !color_only},
        {depth,
         1,
         pixel_offset * task.pass_stride + task.pass_denoising_data + DENOISING_PASS_DEPTH,
         false,
         !color_only},
    };

    for (int i = 0; i < sizeof(passes) / sizeof(passes[0]); i++) {
      if (!passes[i].use) {
        continue;
      }

      passes[i].vector.alloc(w * passes[i].num_elements, h, 1);
#  if 0
      float *host_pointer = (float*)passes[i].vector.host_pointer;
      if (passes[i].scale && scale != 1.0f) {
        /* Normalize albedo and normal passes as they are scaled by the number of samples.
         * For the color passes OIDN will perform auto-exposure making it unnecessary. */

        for (int y = 0; y < h; y++) {
          const float *pass_row = buffer.data() + passes[i].offset + y * row_stride;
          float *scaled_row = host_pointer + y * w * passes[i].num_elements;

          for (int x = 0; x < w; x++) {
            for (int e = 0; e < passes[i].num_elements; e++) {
              scaled_row[x * passes[i].num_elements + e] = pass_row[x * pixel_stride + e] * scale;
            }
          }
        }
      }
      else {
        for (int y = 0; y < h; y++) {
          const float *pass_row = buffer.data() + passes[i].offset + y * row_stride;
          float *row = host_pointer + y * w * passes[i].num_elements;

          for (int x = 0; x < w; x++) {
            for (int e = 0; e < passes[i].num_elements; e++) {
              row[x * passes[i].num_elements + e] = pass_row[x * pixel_stride + e];
            }
          }
        }
      }
#endif
      passes[i].vector.copy_to_device();
    }
#if 1
    device_vector<float> input(this, "tile data", MemoryType::MEM_READ_WRITE);
    input.steal_data(buffer);
    input.copy_to_device();

    cl_kernel filter_split_aov = denoising_program(ustring("filter_split_aov"));

    int arg_ofs = 0;
    arg_ofs += kernel_set_args(
        filter_split_aov, arg_ofs, input.device_pointer, pixel_stride, row_stride);
    
    for (int i = 0; i < sizeof(passes) / sizeof(passes[0]); i++) {
      arg_ofs += kernel_set_args(
          filter_split_aov, arg_ofs, passes[i].vector.device_pointer, passes[i].offset);
    }

    arg_ofs += kernel_set_args(
        filter_split_aov, arg_ofs, w, h, scale, color_only);

    enqueue_kernel(filter_split_aov, w, h);
#endif
  }

  bool launch_denoise(DeviceTask &task, RenderTile &rtile, DenoisingTask &denoising)
  {
    // Update current sample (for display and NLM denoising task)
    rtile.sample = rtile.start_sample + rtile.num_samples;

    // Choose between RIF and NLM denoising
    if (task.denoising.type == DENOISER_RIF) {
      /* Per-tile denoising. */
      const float scale = 1.0f / rtile.sample;
      const float invscale = rtile.sample;
      const int pass_stride = task.pass_stride;

      /* Map neighboring tiles into one buffer for denoising. */
      RenderTileNeighbors neighbors(rtile);
      task.map_neighbor_tiles(neighbors, this);
      RenderTile &center_tile = neighbors.tiles[RenderTileNeighbors::CENTER];
      rtile = center_tile;

      /* Calculate size of the tile to denoise (including overlap). The overlap
       * size was chosen empirically. OpenImageDenoise specifies an overlap size
       * of 128 but this is significantly bigger than typical tile size. */
      int4 rect = center_tile.bounds();  // rect_clip(rect_expand(center_tile.bounds(), 64),
                                         // neighbors.bounds());
      const int2 rect_size = make_int2(rect.z - rect.x, rect.w - rect.y);

      array<float> merged;
      merge_tiles(task, scale, neighbors, merged, rect, rect_size, pass_stride);

      const int color_only = task.denoising.input_passes < DENOISER_INPUT_RGB_ALBEDO_NORMAL;

      device_vector<float> color(this, "color buffer", MemoryType::MEM_READ_WRITE);
      device_vector<float> albedo(this, "albedo buffer", MemoryType::MEM_READ_WRITE);
      /* TODO: map normals from 3 to 2 components*/
      device_vector<float> normals(this, "normals buffer", MemoryType::MEM_READ_WRITE);
      device_vector<float> depth(this, "depth buffer", MemoryType::MEM_READ_WRITE);
      split_aov(task,
                merged,
                0,
                rect_size.x,
                0,
                0,
                rect_size.x,
                rect_size.y,
                1.0f,
                color_only,
                color,
                albedo,
                normals,
                depth);

      rif_image_desc desc = {0};

      desc.image_width = rect_size.x;
      desc.image_height = rect_size.y;
      desc.image_depth = 1;
      desc.num_components = 3;
      desc.type = RIF_COMPONENT_TYPE_FLOAT32;

      device_vector<float> output(this, "output buffer", MemoryType::MEM_READ_WRITE);
      output.alloc(size_t(desc.image_width * desc.num_components),
                   size_t(desc.image_height),
                   size_t(desc.image_depth));
      output.copy_to_device();

      rif_object<rif_image> output_img;
      check_result_rif_ret(rifContextCreateImageFromOpenClMemory(
          context, &desc, CL_MEM_PTR(output.device_pointer), &output_img));

      rif_object<rif_image> color_img;
      check_result_rif_ret(rifContextCreateImageFromOpenClMemory(
          context, &desc, CL_MEM_PTR(color.device_pointer), &color_img));

      rif_object<rif_image> albedo_img;
      if (!color_only) {
        check_result_rif_ret(rifContextCreateImageFromOpenClMemory(
            context, &desc, CL_MEM_PTR(albedo.device_pointer), &albedo_img));
      }

      rif_object<rif_image> normals_img;
      if (!color_only) {
        check_result_rif_ret(rifContextCreateImageFromOpenClMemory(
            context, &desc, CL_MEM_PTR(normals.device_pointer), &normals_img));
        check_result_rif_ret(rifCommandQueueAttachImageFilter(
            queue, remap_normal_filter, normals_img, normals_img));
      }

      desc.num_components = 1;
      rif_object<rif_image> depth_img;
      if (!color_only) {
        check_result_rif_ret(rifContextCreateImageFromOpenClMemory(
            context, &desc, CL_MEM_PTR(depth.device_pointer), &depth_img));
        check_result_rif_ret(
            rifCommandQueueAttachImageFilter(queue, remap_depth_filter, depth_img, depth_img));
      }

      check_result_rif_ret(rifImageFilterSetParameterImage(denoise_filter, "colorImg", color_img));
      if (color_only) {
        check_result_rif_ret(
            rifImageFilterClearParameterImage(denoise_filter, "albedoImg"));
        check_result_rif_ret(
            rifImageFilterClearParameterImage(denoise_filter, "normalsImg"));
        check_result_rif_ret(
            rifImageFilterClearParameterImage(denoise_filter, "depthImg"));
      }
      else {
        check_result_rif_ret(
            rifImageFilterSetParameterImage(denoise_filter, "albedoImg", albedo_img));
        check_result_rif_ret(
            rifImageFilterSetParameterImage(denoise_filter, "normalsImg", normals_img));
        check_result_rif_ret(
            rifImageFilterSetParameterImage(denoise_filter, "depthImg", depth_img));
      }

      check_result_rif_ret(
          rifCommandQueueAttachImageFilter(queue, denoise_filter, color_img, output_img));
      check_result_rif_ret(rifContextExecuteCommandQueue(context, queue, nullptr, nullptr, nullptr));
      check_result_rif_ret(rifCommandQueueDetachImageFilter(queue, denoise_filter));

      if (!color_only) {
        check_result_rif_ret(rifCommandQueueDetachImageFilter(queue, remap_normal_filter));
        check_result_rif_ret(rifCommandQueueDetachImageFilter(queue, remap_depth_filter));
      }
      /* Copy back result from merged buffer. */
      RenderTile &target = neighbors.target;
      const int xmin = max(target.x, rect.x);
      const int ymin = max(target.y, rect.y);
      const int xmax = min(target.x + target.w, rect.z);
      const int ymax = min(target.y + target.h, rect.w);

#  if 0
      float *data = nullptr;
      check_result_rif_ret(rifImageMap(output_img, RIF_IMAGE_MAP_READ, (void **)&data));

      cl_mem target_mem = CL_MEM_PTR(target.buffer);
      cl_int result;
      float *target_data = (float *)clEnqueueMapBuffer(cqCommandQueue,
                                                       target_mem,
                                                       CL_TRUE,
                                                       CL_MAP_WRITE,
                                                       0,
                                                       target.device_size,
                                                       0,
                                                       nullptr,
                                                       nullptr,
                                                       &result);
      opencl_assert_err(result, "clEnqueueMapBuffer");

      for (int y = ymin; y < ymax; y++) {
        float *target_row = target_data + pass_stride * target.offset + y * pass_stride * target.stride;
        const float *data_row = data + (y - ymin) * rect_size.x * 3;

        for (int x = xmin; x < xmax; x++) {
          target_row[pass_stride * x + 0] = data_row[3 * (x - xmin) + 0] * invscale;
          target_row[pass_stride * x + 1] = data_row[3 * (x - xmin) + 1] * invscale;
          target_row[pass_stride * x + 2] = data_row[3 * (x - xmin) + 2] * invscale;
        }
      }
      opencl_assert(
          clEnqueueUnmapMemObject(cqCommandQueue, target_mem, target_data, 0, nullptr, nullptr));

      check_result_rif_ret(rifImageUnmap(output_img, data));
#else
      cl_kernel filter_write_color = denoising_program(ustring("filter_write_color"));

      int arg_ofs = 0;
      arg_ofs += kernel_set_args(filter_write_color,
                                 arg_ofs,
                                 output.device_pointer,
                                 pass_stride,
                                 target.offset,
                                 target.stride);

      arg_ofs += kernel_set_args(filter_write_color, arg_ofs, target.buffer, rect_size.x);
      arg_ofs += kernel_set_args(filter_write_color, arg_ofs, xmin, xmax, ymin, ymax, invscale);

      enqueue_kernel(filter_write_color, size_t(xmax - xmin), size_t(ymax - ymin));
#  endif
      task.unmap_neighbor_tiles(neighbors, this);
    }
    else {
      // Run OpenCL denoising kernels
      DenoisingTask denoising(this, task);
      OpenCLDevice::denoise(rtile, denoising);
    }

    // Update task progress after the denoiser completed processing
    task.update_progress(&rtile, rtile.w * rtile.h);

    return true;
  }
};

bool device_rif_init()
{
  // Need to initialize OpenCL as well
  if (!device_opencl_init())
    return false;

  rif_int device_count = 0;
  rif_int result = rifGetDeviceCount(RIF_BACKEND_API_OPENCL, &device_count);

  if (result != RIF_SUCCESS) {
    VLOG(1) << "RIF initialization failed with error code " << (unsigned int)result << ": "
            << rifGetLastErrorMessage();
    return false;
  }
  if (device_count == 0) {
    VLOG(1) << "RIF initialization failed. No device found.";
    return false;
  }

  // Loaded RIF successfully!
  return true;
}

void device_rif_info(const vector<DeviceInfo> &opencl_devices, vector<DeviceInfo> &devices)
{
  devices.reserve(opencl_devices.size());

  // Simply add all supported OpenCL devices as RIF devices again
  for (DeviceInfo info : opencl_devices) {
    assert(info.type == DEVICE_OPENCL);

    info.type = DEVICE_RIF;
    info.id += "_RIF";
    info.denoisers |= DENOISER_RIF;

    devices.push_back(info);
  }
}

Device *device_rif_create(DeviceInfo &info, Stats &stats, Profiler &profiler, bool background)
{
  return new RIFDevice(info, stats, profiler, background);
}

CCL_NAMESPACE_END

#endif
