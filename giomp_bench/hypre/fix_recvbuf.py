import sys
p = sys.argv[1] + '/src/struct_mv/struct_communication.c'
s = open(p).read()
old_fn = s[s.index('HYPRE_Complex *\nhypre_StructCommunicationGetBuffer'):s.index('HYPRE_Int\nhypre_StructCommunicationReleaseBuffer')]
new_fn = '''HYPRE_Complex *
hypre_StructCommunicationGetBuffer(HYPRE_MemoryLocation memory_location,
                                   HYPRE_Int            size,
                                   HYPRE_Int            recv)
{
   HYPRE_Complex *ptr;

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      /* Separate send and recv buffers: one shared buffer would let the
       * receive overwrite data still being sent. */
      HYPRE_Complex **buf  = recv ? &hypre_HandleStructCommRecvBuffer(hypre_handle()) :
                             &hypre_HandleStructCommSendBuffer(hypre_handle());
      HYPRE_Int      *bufsize = recv ? &hypre_HandleStructCommRecvBufferSize(hypre_handle()) :
                                &hypre_HandleStructCommSendBufferSize(hypre_handle());
      if (size > *bufsize)
      {
         HYPRE_Int new_size = 5 * size;
         *bufsize = new_size;
         hypre_TFree(*buf, memory_location);
         *buf = hypre_CTAlloc(HYPRE_Complex, new_size, memory_location);
      }

      ptr = *buf;
   }
   else
#endif
   {
      ptr = hypre_CTAlloc(HYPRE_Complex, size, memory_location);
   }

   return ptr;
}

'''
s = s.replace(old_fn, new_fn)
a = 'send_buffers[0] = hypre_StructCommunicationGetBuffer(memory_location, size);'
b = 'recv_buffers[0] = hypre_StructCommunicationGetBuffer(memory_location, size);'
assert a in s and b in s
s = s.replace(a, a.replace('size);', 'size, 0);')).replace(b, b.replace('size);', 'size, 1);'))
open(p, 'w').write(s)
print('patched', p)
